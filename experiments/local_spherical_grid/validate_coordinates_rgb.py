"""冻结 Yin/Yang 的坐标及真实 RGB 映射链路验证；不搜索、不修改模型或 GT。"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from camera_geometry import load_pair, pair_geometry, project_camera, sample_rgb, OCAM_FROM_VIS
from search_yang_grid import angular_coordinates, native_rays, window_mask, rx, ry
from dataset_stereo import feature_grid


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def angle_error(a, b):
    a, b = unit(a), unit(b)
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a*b, axis=-1)))


def summary(x):
    x = np.asarray(x)
    return dict(count=int(x.size), mean=float(np.mean(x)) if x.size else None,
                p95=float(np.percentile(x, 95)) if x.size else None,
                p99=float(np.percentile(x, 99)) if x.size else None,
                maximum=float(np.max(x)) if x.size else None)


def through_chart(r4, basis):
    # 行向量约定：r4=rV @ A.T；rChart=rV @ Q；逆变换 rV=rChart @ Q.T。
    visual = r4 @ OCAM_FROM_VIS
    local = visual @ basis
    lam = np.arctan2(local[:, 0], local[:, 2])
    phi = np.arcsin(np.clip(local[:, 1], -1., 1.))
    rebuilt = np.stack((np.cos(phi)*np.sin(lam), np.sin(phi), np.cos(phi)*np.cos(lam)), -1)
    return rebuilt @ basis.T @ OCAM_FROM_VIS.T


def camera_pixels_to_reference(camera_index, pixels, cameras, rotation, translation, radius):
    rays_cam = unit(cameras[camera_index].pixelToRay(pixels).T)
    if camera_index == 0:
        return radius*rays_cam, np.isfinite(rays_cam).all(1)
    # cam3 光心在 cam4 中不是零点。取右像素射线与 cam4-centred 球面的正交点。
    # X4=C3+s*d4, ||X4||=radius；radius>baseline 时相机在球内，只有一个正根。
    center = -rotation.T @ translation
    direction = rays_cam @ rotation
    dot = direction @ center
    disc = dot*dot + radius*radius - center@center
    distance = -dot + np.sqrt(np.maximum(disc, 0))
    points = center + distance[:, None]*direction
    return points, np.isfinite(points).all(1) & (disc >= 0) & (distance > 0)


def project_points(camera_index, points4, cameras, rotation, translation):
    points = points4.T if camera_index == 0 else rotation @ points4.T+translation[:, None]
    return project_camera(cameras[camera_index], points)


def bilinear_array(values, valid_grid, u, v):
    """只使用正权重角点；strict 支持不完整时保持无效，不作边界复制或补洞。"""
    h, w = valid_grid.shape
    inside = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u <= w-1) & (v >= 0) & (v <= h-1)
    safe_u, safe_v = np.clip(np.nan_to_num(u), 0, w-1), np.clip(np.nan_to_num(v), 0, h-1)
    x0, y0 = np.floor(safe_u).astype(int), np.floor(safe_v).astype(int)
    x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
    dx, dy = safe_u-x0, safe_v-y0
    weights = [(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy]
    indices = [(y0, x0), (y0, x1), (y1, x0), (y1, x1)]
    strict = inside.copy()
    output = np.zeros((len(u), values.shape[-1]), dtype=np.float64)
    for weight, (yy, xx) in zip(weights, indices):
        strict &= (weight <= 1e-12) | valid_grid[yy, xx]
        output += weight[:, None]*values[yy, xx]
    output[~strict] = 0
    return output, strict


def chart_sample(image, source_camera_index, points4, basis, bounds, native_size, camera, rotation, translation):
    # 原图规则网格含无效角点，首个点可能是 NaN；半径取有效点的径向范数。
    radius = float(np.nanmedian(np.linalg.norm(points4, axis=-1)))
    rays, _ = native_rays(bounds, basis, native_size)
    native_points = rays.reshape(-1, 3).T*radius
    if source_camera_index:
        native_points = rotation@native_points+translation[:, None]
    native_rgb, native_valid, native_uv = sample_rgb(camera, image, native_points, native_size)
    rvis = unit(points4) @ OCAM_FROM_VIS
    lon, lat = angular_coordinates(rvis, basis)
    delta = (lon-bounds['longitude_center_deg']+180) % 360-180
    u = (delta+bounds['longitude_half_span_deg'])/(2*bounds['longitude_half_span_deg'])*(native_size[1]-1)
    v = (bounds['latitude_max_deg']-lat)/(bounds['latitude_max_deg']-bounds['latitude_min_deg'])*(native_size[0]-1)
    recovered, strict = bilinear_array(native_rgb.astype(float), native_valid, u, v)
    recovered_uv, strict_uv = bilinear_array(np.moveaxis(native_uv, 0, -1), native_valid, u, v)
    strict &= strict_uv & window_mask(lon, lat, bounds)
    recovered[~strict] = 0
    return recovered, recovered_uv, strict, native_rgb, native_valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--selected-json', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--frames', nargs='+', type=int, default=[4887, 4888, 4889])
    parser.add_argument('--random-directions', type=int, default=100000)
    parser.add_argument('--random-pixels', type=int, default=12000)
    parser.add_argument('--seed', type=int, default=20261114)
    args = parser.parse_args()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    selected = json.loads(args.selected_json.read_text())
    chart_bases = {'yin': np.eye(3), 'yang': np.asarray(selected['yang_basis_visual'], dtype=np.float64)}
    chart_bounds = {'yin': selected['yin_bounds'], 'yang': selected['yang_bounds']}
    native_sizes = {name: tuple(selected['native_sampling'][name]['shape_dhw'][1:]) for name in chart_bases}
    rng = np.random.default_rng(args.seed)
    random_directions = unit(rng.normal(size=(args.random_directions, 3)))
    result = dict(seed=args.seed, random_directions=args.random_directions, frames=args.frames,
                  frozen_parameters_sha256=hashlib.sha256(args.selected_json.read_bytes()).hexdigest(),
                  bases={name: basis.tolist() for name, basis in chart_bases.items()},
                  random_direction_roundtrips={}, pixel_roundtrips=[], rgb_roundtrips=[],
                  rgb_overlap_consistency=[], frame_geometry_hashes={})
    for name, basis in chart_bases.items():
        restored = through_chart(random_directions, basis)
        result['random_direction_roundtrips'][name] = summary(angle_error(random_directions, restored))
        assert np.max(angle_error(random_directions, restored)) < 1e-9
    # 复核公开角度与所保存矩阵：矩阵右端先作用于列向量。
    winner = selected['selected']
    expected = ry(winner['yaw_y_deg'])@rx(winner['tilt_x_deg'])@ry(winner['longitude_center_deg'])
    result['euler_reconstruction_max'] = float(np.max(np.abs(expected-chart_bases['yang'])))
    result['basis_determinants'] = {name: float(np.linalg.det(basis)) for name, basis in chart_bases.items()}
    assert result['euler_reconstruction_max'] < 1e-12
    axis = np.asarray(selected['inverse_depth_candidates'])
    selected_indices = [48, 95]
    arrays = dict(random_directions_reference=random_directions)
    old_native_hashes = None
    plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'savefig.dpi': 160})
    for frame in args.frames:
        cameras, images, config, provenance = load_pair(args.data_root, 4, 3, frame)
        rotation, translation, geometry = pair_geometry(*cameras)
        if frame == args.frames[0]:
            result['cnn_feature_grid_diagnostics'] = []
            for name, basis in chart_bases.items():
                native, _ = native_rays(chart_bounds[name], basis, native_sizes[name])
                for depth_index in [0, 48, 95]:
                    for source_index, camera in enumerate(cameras):
                        points = native.reshape(-1, 3).T / axis[depth_index]
                        if source_index: points = rotation@points+translation[:, None]
                        uv, raw_valid = project_camera(camera, points)
                        grid, cnn_valid = feature_grid(camera, uv, raw_valid)
                        hf, wf = (camera.height+1)//2, (camera.width+1)//2
                        correct = (uv/2).T
                        naive = np.stack((uv[0]*(wf-1)/(camera.width-1), uv[1]*(hf-1)/(camera.height-1)), 1)
                        recovered = (grid[cnn_valid]+1)*np.array([wf-1, hf-1])/2
                        result['cnn_feature_grid_diagnostics'].append(dict(chart=name, camera=int(camera.id),
                            depth_index=depth_index, raw_valid_count=int(raw_valid.sum()), cnn_valid_count=int(cnn_valid.sum()),
                            raw_valid_but_cnn_invalid_count=int(np.count_nonzero(raw_valid & ~cnn_valid)),
                            inverse_coordinate_error_feature_px=float(np.max(np.abs(recovered-correct[cnn_valid]))),
                            naive_image_resize_bias_feature_px=summary(np.linalg.norm(naive[raw_valid]-correct[raw_valid], axis=1)),
                            feature_hw=[hf,wf], raw_centres_supported_uv_max=[2*(wf-1),2*(hf-1)]))
        hashes = {}
        for source_index, (camera, image) in enumerate(zip(cameras, images)):
            yy, xx = np.meshgrid(np.arange(camera.height), np.arange(camera.width), indexing='ij')
            candidates = np.flatnonzero(~camera.invalid_mask.ravel())
            pick = rng.choice(candidates, min(len(candidates), args.random_pixels), replace=False)
            pixels = np.stack((xx.ravel()[pick], yy.ravel()[pick]))
            for depth_index in selected_indices:
                radius = 1/axis[depth_index]
                points, good_ray = camera_pixels_to_reference(source_index, pixels, cameras, rotation, translation, radius)
                direct_uv, direct_valid = project_points(source_index, points, cameras, rotation, translation)
                other_uv, other_valid = project_points(1-source_index, points, cameras, rotation, translation)
                common = good_ray & direct_valid & other_valid
                direct_error = np.linalg.norm(direct_uv-pixels, axis=0)
                chart_outputs = {}
                for name, basis in chart_bases.items():
                    restored_points = through_chart(unit(points), basis)*radius
                    restored_uv, restored_valid = project_points(source_index, restored_points, cameras, rotation, translation)
                    valid = good_ray & direct_valid & restored_valid
                    chain_error = np.linalg.norm(restored_uv-pixels, axis=0)
                    incremental_error = np.linalg.norm(restored_uv-direct_uv, axis=0)
                    row = dict(frame=frame, camera=int(camera.id), chart=name, depth_index=depth_index,
                        radius_m=float(radius), valid_count=int(valid.sum()),
                        analytic_pixel_ray_chart_pixel_error_px=summary(chain_error[valid]),
                        original_ocam_pixel_ray_pixel_error_px=summary(direct_error[valid]),
                        added_by_chart_max_px=float(incremental_error[valid].max()),
                        direction_error_deg=summary(angle_error(unit(points)[valid], unit(restored_points)[valid])))
                    result['pixel_roundtrips'].append(row)
                    assert row['added_by_chart_max_px'] < 1e-8
                    assert row['analytic_pixel_ray_chart_pixel_error_px']['maximum'] < .01
                    recovered, uv_recovered, strict, native_rgb, native_valid = chart_sample(
                        image, source_index, points, basis, chart_bounds[name], native_sizes[name], camera, rotation, translation)
                    hashes[f'{camera.id}_{name}_{depth_index}'] = hashlib.sha256(native_valid.tobytes()).hexdigest()
                    mask = strict & common
                    chart_outputs[name] = (recovered, uv_recovered, mask)
                    rgb_error = np.mean(np.abs(recovered-image[pixels[1], pixels[0]]), axis=1)
                    uv_error = np.linalg.norm(uv_recovered-pixels.T, axis=1)
                    result['rgb_roundtrips'].append(dict(frame=frame, camera=int(camera.id), chart=name,
                        depth_index=depth_index, common_random_pixels=int(common.sum()),
                        strict_native_support_count=int(mask.sum()),
                        rgb_mae_0_255=summary(rgb_error[mask]),
                        native_uv_field_reconstruction_error_px=summary(uv_error[mask]),
                        note='RGB native map uses original OpenCV INTER_LINEAR; inverse native lookup uses exact bilinear weights. '
                             'Finite sampling error is not an OCam or extrinsics error.'))
                    tag = f'f{frame}_cam{camera.id}_{name}_idx{depth_index}'
                    Image.fromarray(native_rgb).save(out/f'{tag}_native_rgb.png')
                    arrays[tag+'_source_pixels_uv'] = pixels.astype(np.int16)
                    arrays[tag+'_analytic_return_pixels_uv'] = restored_uv
                    arrays[tag+'_native_return_rgb'] = recovered.astype(np.float32)
                    arrays[tag+'_strict_common_support'] = mask
                    arrays[tag+'_rgb_error'] = rgb_error.astype(np.float32)

                overlap = chart_outputs['yin'][2] & chart_outputs['yang'][2]
                overlap_rgb_difference = np.mean(np.abs(chart_outputs['yin'][0]-chart_outputs['yang'][0]), axis=1)
                overlap_uv_difference = np.linalg.norm(chart_outputs['yin'][1]-chart_outputs['yang'][1], axis=1)
                result['rgb_overlap_consistency'].append(dict(frame=frame, camera=int(camera.id), depth_index=depth_index,
                    strict_overlap_count=int(overlap.sum()),
                    same_camera_yin_yang_rgb_mae=summary(overlap_rgb_difference[overlap]),
                    same_camera_yin_yang_uv_difference_px=summary(overlap_uv_difference[overlap]),
                    note='Independent finite native lattices resample the same RGB with different phases; '
                         'nonzero RGB differences are not cross-camera or GT visibility errors.'))

                # 首帧全图每两像素复查，为看清翻转/接缝/空洞保留全尺寸原图坐标。
                if frame == args.frames[0]:
                    grid_u, grid_v = np.meshgrid(np.arange(0, camera.width, 2), np.arange(0, camera.height, 2))
                    shape = grid_u.shape
                    grid_pixels = np.stack((grid_u.ravel(), grid_v.ravel()))
                    grid_points, grid_valid = camera_pixels_to_reference(source_index, grid_pixels, cameras, rotation, translation, radius)
                    _, self_valid = project_points(source_index, grid_points, cameras, rotation, translation)
                    _, other_valid = project_points(1-source_index, grid_points, cameras, rotation, translation)
                    physical_common = grid_valid & self_valid & other_valid
                    choices = []
                    for name, basis in chart_bases.items():
                        rgb, uv, strict, _, _ = chart_sample(image, source_index, grid_points, basis, chart_bounds[name],
                                                           native_sizes[name], camera, rotation, translation)
                        _, lat = angular_coordinates(unit(grid_points)@OCAM_FROM_VIS, basis)
                        kappa = 1/np.maximum(np.cos(np.deg2rad(lat)), 1e-6)
                        choices.append((rgb, uv, strict & physical_common, kappa))
                    owner = (np.where(choices[0][2], choices[0][3], np.inf) <=
                             np.where(choices[1][2], choices[1][3], np.inf))
                    supported = choices[0][2] | choices[1][2]
                    restored_rgb = np.where(owner[:, None], choices[0][0], choices[1][0])
                    direct = image[grid_v, grid_u].reshape(-1, 3).astype(float)
                    rgb_error = np.mean(np.abs(restored_rgb-direct), axis=1)
                    display = np.full((len(owner), 3), 235., dtype=float)
                    display[supported] = restored_rgb[supported]
                    codes = np.zeros(len(owner), np.uint8)
                    codes[physical_common] = 1
                    codes[supported] = 2
                    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), layout='constrained')
                    axes[0].imshow(image[::2, ::2]); axes[0].set_title(f'cam{camera.id} original RGB')
                    axes[1].imshow(np.clip(display.reshape(*shape, 3), 0, 255).astype('uint8')); axes[1].set_title('Fisheye -> grids -> fisheye')
                    im = axes[2].imshow(np.ma.array(rgb_error.reshape(shape), mask=~supported.reshape(shape)),
                                        cmap='magma', vmin=0, vmax=40)
                    axes[2].set_title('RGB MAE; display clipped at 40')
                    fig.colorbar(im, ax=axes[2], fraction=.04, label='Mean absolute error (0-255)')
                    axes[3].imshow(codes.reshape(shape), cmap=matplotlib.colors.ListedColormap(['#e8ebf0', '#cc3c41', '#0072b2']), vmin=0, vmax=2)
                    axes[3].set_title('Blue support; red missing support')
                    for ax in axes: ax.set_xlabel('Fisheye column / 2'); ax.set_ylabel('Fisheye row / 2')
                    fig.suptitle(f'Frame {frame}, candidate {depth_index}, cam4-centred radius {radius:.3f} m')
                    fig.savefig(out/f'cam{camera.id}_idx{depth_index}_rgb_roundtrip.png'); plt.close(fig)
                    arrays[f'cam{camera.id}_idx{depth_index}_physical_common'] = physical_common.reshape(shape)
                    arrays[f'cam{camera.id}_idx{depth_index}_native_supported'] = supported.reshape(shape)
                    arrays[f'cam{camera.id}_idx{depth_index}_recovered_rgb'] = display.reshape(*shape, 3).astype(np.float32)
                    arrays[f'cam{camera.id}_idx{depth_index}_rgb_mae'] = rgb_error.reshape(shape).astype(np.float32)
                    both = choices[0][2] & choices[1][2]
                    diff = np.mean(np.abs(choices[0][0]-choices[1][0]), axis=1)
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout='constrained')
                    for i in range(2):
                        display = np.full((len(owner), 3), 235.)
                        display[both] = choices[i][0][both]
                        axes[i].imshow(np.clip(display.reshape(*shape,3),0,255).astype('uint8'))
                        axes[i].set_title(['Yin return RGB on strict overlap', 'Yang return RGB on strict overlap'][i])
                    im=axes[2].imshow(np.ma.array(diff.reshape(shape),mask=~both.reshape(shape)),cmap='magma',vmin=0,vmax=30)
                    axes[2].set_title('Same-source chart disagreement')
                    fig.colorbar(im,ax=axes[2],fraction=.04,label='Mean absolute RGB difference')
                    fig.suptitle(f'cam{camera.id}, candidate {depth_index}: finite lattice interpolation, not stereo error')
                    fig.savefig(out/f'cam{camera.id}_idx{depth_index}_overlap_rgb.png');plt.close(fig)
                    arrays[f'cam{camera.id}_idx{depth_index}_same_source_overlap_rgb_mae'] = diff.reshape(shape).astype(np.float32)
                    arrays[f'cam{camera.id}_idx{depth_index}_same_source_overlap_valid'] = both.reshape(shape)
        result['frame_geometry_hashes'][str(frame)] = hashes
        if old_native_hashes is not None:
            assert hashes == old_native_hashes, 'Geometry changed with RGB frame content'
        old_native_hashes = hashes
    np.savez_compressed(out/'coordinates_rgb_diagnostics.npz', **arrays)
    result['geometry_masks_identical_across_frames'] = True
    result['status'] = 'PASS'
    result['limitations'] = ['Roundtrip sampling does not test stereo photometric consistency on a true scene surface.',
        'Finite native-grid interpolation smooths high frequencies; large RGB residuals at object edges do not by themselves imply wrong geometry.',
        'Strict native support excludes invalid positive-weight neighbours rather than filling them.',
        'No depth GT was used in this program.']
    (out/'coordinates_rgb_result.json').write_text(json.dumps(result, indent=2)+'\n')
    max_ray = max(x['maximum'] for x in result['random_direction_roundtrips'].values())
    max_pixel = max(x['analytic_pixel_ray_chart_pixel_error_px']['maximum'] for x in result['pixel_roundtrips'])
    added = max(x['added_by_chart_max_px'] for x in result['pixel_roundtrips'])
    text = ['# 坐标与真实 RGB 链路验证', '', '## 坐标与矩阵约定', '',
        '所有球心均为 cam4 光心。真实 cam4/OCam 轴为 x 向图像右、y 向下、z 沿左光轴。'
        'Yin 采用显示坐标：x 向图像左、y 向上、z 沿左光轴；其相对 cam4 的旋转为 A=diag(-1,-1,1)。',
        'Yang 的 Q=R_visual_from_Yang；其每一列是 Yang 的一个坐标轴在 Yin/visual 系中的坐标，'
        '不是投影矩阵，也不包含基线平移。', '',
        '列向量：r4=A rV，rV=A^T r4；rV=Q rYang，rYang=Q^T rV。',
        '行向量批量数组：rV=r4 @ A，rYang=rV @ Q；逆向 rV=rYang @ Q.T，r4=rV @ A.T。',
        'Q=Ry(yaw_y) @ Rx(tilt_x) @ Ry(原最短经度弧中心)。右端矩阵先作用于列向量；'
        '最后一项把原经度中心吸收到最终旋转，因此冻结后的 Yang 经度中心为 0°。', '',
        '```json', json.dumps(result['bases'], indent=2), '```', '',
        '## 实际测试', '', f'固定种子 {args.seed}，{args.random_directions} 条随机单位球面方向；'
        '角误差使用 atan2(||a×b||, a·b)，不使用近 1 时不稳定的 arccos。',
        f'随机方向最大角误差 {max_ray:.9g} 度。真实有效鱼眼像素完整解析链路最大误差 {max_pixel:.9g} px，'
        f'其中相对于直接 OCam 往返，网格坐标变换新增误差最大 {added:.9g} px。',
        'cam3 随机像素先使用真实基线求其射线与 cam4-centred 有限球面的交点，然后经过完整 R,t 投回 cam3；没有把两光心强行重合。',
        f'帧 {args.frames} 的原生投影有效性 Mask 哈希完全相同，几何不依赖 RGB 内容。', '',
        '## 原生网格 RGB 往返', '',
        '每个相机分别做 原鱼眼 RGB→原生 Yin/Yang→原鱼眼像素 的双线性往返，不将两相机在虚拟球面上的 RGB 当成真实场景光度一致性。'
        '只在共同有效且全部正权重角点有效时统计 RGB/UV 误差；不补洞。'
        '另对原生 UV 坐标场进行相同采样，用像素误差检查方向、翻转和位置偏差。', '',
        '**计数口径：** `common_random_pixels` 是双目共同有效的随机源像素数，其中部分方向可能不在当前单一 chart 的有限窗口内。'
        '`strict_native_support_count` 同时受当前 chart 窗口和有效插值角点约束。'
        '因此表中 strict/common 比值不能直接解释为插值支持率或空洞率；两者并非仅差插值支持这一条件。'
        '全域采样支持率及空洞率以主验证程序在相同参考方向集合上的多分辨率检查为准。', '',
        '| frame | cam | chart | idx | strict/common pixels | RGB MAE | RGB P99 | UV mean px | UV max px |',
        '|---|---|---|---:|---|---:|---:|---:|---:|']
    for row in result['rgb_roundtrips']:
        rgb, uv = row['rgb_mae_0_255'], row['native_uv_field_reconstruction_error_px']
        text.append(f"|{row['frame']}|{row['camera']}|{row['chart']}|{row['depth_index']}|{row['strict_native_support_count']}/{row['common_random_pixels']}|"
                    f"{rgb['mean']:.4f}|{rgb['p99']:.4f}|{uv['mean']:.5f}|{uv['maximum']:.5f}|")
    text += ['', '## 同相机两网格重叠区 RGB 一致性', '',
        '下面比较同一个源相机、同一个球面点经两套不同原生格点重采样后的 RGB。'
        '其差异量化了潜在的颜色接缝；它不是左右相机的光度误差，也不是遮挡真值。'
        '即使解析坐标正确，不同格点相位与有限分辨率仍会在高频纹理和物体边缘产生差异。', '',
        '| frame | cam | idx | strict overlap | RGB difference mean | RGB difference P99 | UV difference max px |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in result['rgb_overlap_consistency']:
        rgb,uv=row['same_camera_yin_yang_rgb_mae'],row['same_camera_yin_yang_uv_difference_px']
        text.append(f"|{row['frame']}|{row['camera']}|{row['depth_index']}|{row['strict_overlap_count']}|{rgb['mean']:.4f}|"
                    f"{rgb['p99']:.4f}|{uv['maximum']:.6f}|")
    text += ['', '原生网格有限分辨率导致的抗混叠/边界重采样误差与解析坐标误差应分开解读。'
        '红色缺少支持表示离散邻点 Mask 不完整，不能据 RGB 看起来平滑就宣称无空洞。', '',
        '## 原图→CNN 特征坐标只读核查', '',
        '原图 768×800，原 Encoder stride=2 输出 384×400；特征中心 j 对应原图 2*j。'
        '因此先 u_feature=u_raw/2，再用 align_corners=True 归一化：g_x=2*u_feature/(Wf-1)-1。'
        '不能直接用 u_raw/(Wraw-1) 作为特征坐标比例。最后一列特征中心为原图 u=798、最后一行为 v=766，'
        '原图中投影合法的边缘点仍可能落到特征图范围之外。下面调用的是现有 feature_grid，未修改它。', '',
        '| chart | cam | idx | raw valid | CNN valid | raw-only | naive resize bias max (feature px) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in result['cnn_feature_grid_diagnostics']:
        text.append(f"|{row['chart']}|{row['camera']}|{row['depth_index']}|{row['raw_valid_count']}|{row['cnn_valid_count']}|"
                    f"{row['raw_valid_but_cnn_invalid_count']}|{row['naive_image_resize_bias_feature_px']['maximum']:.6f}|")
    text += ['', '这里只核对索引和有效性；没有运行 Encoder，不代表已经验证 CNN 特征语义或后续 Cost Volume。', '',
        '## 输出', '', '- `coordinates_rgb_result.json`：全部实际指标、坐标约定与帧稳定性。',
        '- `coordinates_rgb_diagnostics.npz`：随机像素、解析返回像素、支持 Mask 与 RGB 误差。',
        '- `cam*_idx*_rgb_roundtrip.png`：全图往返、误差与不完整支持位置。',
        '- `*_native_rgb.png`：两相机各自投影至两网格的 RGB，含真实视场外区域。', '',
        '没有使用 GT，没有修改参数或模型；此项验证仅证明坐标链路数值一致及可复查的采样行为。']
    (out/'REPORT_COORDINATES_RGB.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(status=result['status'], maximum_random_angle_deg=max_ray,
                         maximum_analytic_pixel_error_px=max_pixel, maximum_added_chart_pixel_error_px=added,
                         geometry_masks_identical_across_frames=True), indent=2), flush=True)


if __name__ == '__main__':
    main()
