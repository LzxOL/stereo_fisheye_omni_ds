"""纯几何工具：复用工程 OCam / 外参定义，不创建 Dataset 或网络。"""
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.ocam import OcamModel
from utils.geometry import getRot, getTr, applyTransform, mergedTransform
from dataset_stereo import project_camera


# 实际 OCam 为 x 向图像右、y 向下、z 沿光轴。显示坐标以 y 向上、z 沿左光轴；
# x 因而向图像左。绕 z 旋转 180°，行列式为 +1，不是偷偷反射坐标。
OCAM_FROM_VIS = np.diag([-1., -1., 1.])


def load_pair(data_root, ref_cam=4, target_cam=3, frame=4887):
    data_root = Path(data_root).expanduser().resolve()
    config_path = data_root / 'config.yaml'
    config = yaml.safe_load(config_path.read_text())
    cameras, images, provenance = [], [], []
    for cam_id in (ref_cam, target_cam):
        cfg = next(c for c in config['cameras'] if int(c['cam_id']) == cam_id)
        camera = OcamModel()
        camera.setConfig(cfg)
        mask_path = data_root / cfg['invalid_mask']
        # 只读取现成 Mask；不调用可能自动生成、写回 Mask 的数据加载入口。
        if not mask_path.is_file():
            raise FileNotFoundError(f'Actual invalid mask required: {mask_path}')
        camera.invalid_mask = np.asarray(Image.open(mask_path).convert('L')) > 0
        image_path = data_root / (config['dataset']['img_fmt'] % (cam_id, frame))
        image = np.asarray(Image.open(image_path).convert('RGB'))
        if image.shape[:2] != (camera.height, camera.width):
            raise ValueError(f'Image/calibration mismatch: {image_path}, {image.shape}')
        if camera.invalid_mask.shape != image.shape[:2]:
            raise ValueError(f'Mask/calibration mismatch: {mask_path}')
        cameras.append(camera)
        images.append(image)
        provenance.append(dict(config=cfg, image=str(image_path), mask=str(mask_path),
                               ocam_txt=str(data_root / f'ocam{cam_id}.txt'),
                               ocam_txt_exists=(data_root / f'ocam{cam_id}.txt').exists()))
    if ref_cam == target_cam:
        raise ValueError('Reference and target must differ')
    return cameras, images, config, provenance


def pair_geometry(left, right):
    # pose 是 camera-to-rig；从 cam4 到 cam3 需先到 Rig，再到 cam3。
    transform = mergedTransform(right.rig2cam, left.cam2rig)
    rotation, translation = getRot(transform), getTr(transform).ravel()
    right_center = applyTransform(left.rig2cam, getTr(right.cam2rig)).ravel()
    rig_center = applyTransform(left.rig2cam, np.zeros((3, 1))).ravel()
    right_axis = getRot(left.rig2cam) @ getRot(right.cam2rig)[:, 2]
    angle = np.rad2deg(np.arccos(np.clip(right_axis[2], -1., 1.)))
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, translation
    report = dict(reference_id=int(left.id), target_id=int(right.id),
                  reference_center_rig_m=getTr(left.cam2rig).ravel().tolist(),
                  target_center_rig_m=getTr(right.cam2rig).ravel().tolist(),
                  target_center_reference_m=right_center.tolist(),
                  rig_center_reference_m=rig_center.tolist(),
                  reference_optical_axis_reference=[0., 0., 1.],
                  target_optical_axis_reference=right_axis.tolist(),
                  baseline_m=float(np.linalg.norm(right_center)), optical_axis_angle_deg=float(angle),
                  target_from_reference=matrix.tolist(), ocam_from_visual=OCAM_FROM_VIS.tolist(),
                  convention='pose=[Rodrigues vector, translation], camera-to-rig; units metre; '
                             'OCam x image-right, y image-down, z optical-forward; '
                             'visual x image-left, y up, z reference-forward; longitude positive toward image-left',
                  pixels='0-based (u=column, v=row); config center is (row,column); '
                         'reject out-of-bounds/FOV and any invalid bilinear support; no boundary clamping')
    return rotation, translation, report


def spherical_rays(height, width):
    # 像素中心取样避免极点重复；面积采用每个 ERP 像素的精确立体角，
    # dOmega=dLambda*(sin(phi_top)-sin(phi_bottom))，不是简单像素计数。
    lon = -np.pi + (np.arange(width) + .5) * 2 * np.pi / width
    lat_edges = np.linspace(np.pi / 2, -np.pi / 2, height + 1)
    lat = (lat_edges[:-1] + lat_edges[1:]) / 2
    lam, phi = np.meshgrid(lon, lat)
    rays_vis = np.stack((np.cos(phi)*np.sin(lam), np.sin(phi), np.cos(phi)*np.cos(lam)), -1)
    rays_ocam = rays_vis @ OCAM_FROM_VIS.T
    area = np.broadcast_to((np.sin(lat_edges[:-1])-np.sin(lat_edges[1:]))[:, None] *
                           (2*np.pi/width), (height, width)).copy()
    assert np.isclose(area.sum(), 4*np.pi)
    return rays_ocam, rays_vis, lam, phi, area


def sample_rgb(camera, image, points, shape):
    pixels, valid = project_camera(camera, points)
    # 无效位置使用图像外坐标；不把越界坐标截到边缘再填颜色。
    uv = np.where(valid[None], pixels, -1e5).astype(np.float32)
    rgb = cv2.remap(image, uv[0].reshape(shape), uv[1].reshape(shape), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = valid.reshape(shape)
    rgb[~valid] = 0
    return rgb, valid, pixels.reshape(2, *shape)


def sphere_projection(cameras, images, rays_ocam, rotation, translation, radius=None):
    shape = rays_ocam.shape[:2]
    left_points = rays_ocam.reshape(-1, 3).T
    if radius is not None:
        left_points = radius * left_points
    # Angular 不含平移，仅分析远场方向；Finite 才在左光心球面上投影同一三维点。
    right_points = rotation @ left_points
    if radius is not None:
        right_points = right_points + translation[:, None]
    lrgb, lm, luv = sample_rgb(cameras[0], images[0], left_points, shape)
    rrgb, rm, ruv = sample_rgb(cameras[1], images[1], right_points, shape)
    return dict(left_rgb=lrgb, right_rgb=rrgb, left=lm, right=rm, overlap=lm & rm,
                left_uv=luv, right_uv=ruv, radius_m=radius)


def complementary_coordinates(rays_vis, report, lon_span=240., lat_limit=45.):
    right_axis = OCAM_FROM_VIS.T @ np.array(report['target_optical_axis_reference'])
    bisector = right_axis + np.array([0., 0., 1.])
    bisector[1] = 0
    if np.linalg.norm(bisector) < 1e-8:
        raise ValueError('Opposite optical axes have no unique horizontal bisector')
    forward = bisector / np.linalg.norm(bisector)
    pole = np.cross([0., 1., 0.], forward)
    pole = pole / np.linalg.norm(pole)
    xaxis = np.cross(pole, forward)
    basis = np.stack((xaxis, pole, forward), axis=1)
    assert np.isclose(np.linalg.det(basis), 1.)
    local = rays_vis @ basis
    phi = np.arcsin(np.clip(local[..., 1], -1, 1))
    lon = np.arctan2(local[..., 0], local[..., 2])
    # 有限矩形补偿 Patch；不是完整 360° Yin-Yang。只在 Patch 内允许选 kappa_C。
    patch = (np.abs(np.rad2deg(phi)) <= lat_limit) & (np.abs(np.rad2deg(lon)) <= lon_span/2)
    kappa = 1 / np.maximum(np.sqrt(np.maximum(0., 1-local[..., 1]**2)), 1e-6)
    return basis, phi, lon, patch, kappa


def weighted_summary(values, mask, area):
    values, weights = values[mask], area[mask]
    if not len(values):
        return dict(mean=None, p50=None, p95=None, maximum=None)
    order = np.argsort(values)
    cdf = (np.cumsum(weights[order]) - .5*weights[order]) / weights.sum()
    return dict(mean=float(np.average(values, weights=weights)),
                p50=float(np.interp(.5, cdf, values[order])),
                p95=float(np.interp(.95, cdf, values[order])), maximum=float(values.max()))


def coverage_statistics(item, area, phi, phi_c, patch, kp, kc, threshold):
    left, right, overlap = item['left'], item['right'], item['overlap']
    total = area.sum()
    fraction = lambda mask: float(area[mask].sum() / total)
    conditional = lambda mask: float(area[mask & overlap].sum()/area[overlap].sum()) if overlap.any() else 0.
    high = kp > threshold
    best = np.where(patch, np.minimum(kp, kc), kp)
    # 报告分母：full sphere / overlap / primary-high-overlap，不混为一谈。
    high_common = high & overlap
    result = dict(radius_m=item['radius_m'], left_fraction=fraction(left), right_fraction=fraction(right),
                  overlap_fraction=fraction(overlap), overlap_of_left=fraction(overlap)/fraction(left),
                  left_only_fraction=fraction(left & ~right), right_only_fraction=fraction(right & ~left),
                  uncovered_fraction=fraction(~(left | right)), overlap_latitude_min_deg=None,
                  overlap_latitude_max_deg=None, overlap_longitude_extent_note='See ERP map for wrap/holes',
                  high_primary_of_overlap=conditional(high), patch_of_overlap=conditional(patch),
                  high_best_of_overlap=conditional(best > threshold),
                  high_primary_covered_by_patch=float(area[high_common & patch].sum()/area[high_common].sum())
                    if high_common.any() else 0.,
                  kappa_primary=weighted_summary(kp, overlap, area),
                  kappa_complementary_on_patch=weighted_summary(kc, overlap & patch, area),
                  kappa_best_with_patch=weighted_summary(best, overlap, area),
                  primary_high_abs_latitude_in_complement_deg=weighted_summary(np.abs(np.rad2deg(phi_c)), high_common, area),
                  primary_high_kappa_before=weighted_summary(kp, high_common, area),
                  primary_high_kappa_after_with_patch=weighted_summary(best, high_common, area))
    if overlap.any():
        result['overlap_latitude_min_deg'] = float(np.rad2deg(phi[overlap]).min())
        result['overlap_latitude_max_deg'] = float(np.rad2deg(phi[overlap]).max())
    return result


def validate_geometry(cameras, images, rotation, translation, rays, basis, rays_vis, angular, finite):
    checks = {}
    for camera, image in zip(cameras, images):
        u, v = np.meshgrid(np.arange(0, camera.width, 4), np.arange(0, camera.height, 4))
        pixels = np.stack((u.ravel(), v.ravel()))
        ray = camera.pixelToRay(pixels)
        projected, valid = project_camera(camera, ray)
        valid &= ~camera.invalid_mask[v.ravel(), u.ravel()]
        error = np.linalg.norm(projected[:, valid]-pixels[:, valid], axis=0)
        checks[f'cam{camera.id}_pixel_ray_pixel_error_px'] = dict(
            count=int(valid.sum()), median=float(np.median(error)), p95=float(np.percentile(error, 95)),
            maximum=float(error.max()), note='OCam forward/inverse polynomials are approximate inverses')
        assert np.isfinite(error).all() and np.percentile(error, 95) < 2., 'OCam coefficient/order check failed'
        center = np.array([[camera.yc], [camera.xc]])
        axis = camera.pixelToRay(center).ravel()
        axis /= np.linalg.norm(axis)
        assert np.allclose(axis, [0, 0, 1])

    point = (2*rays.reshape(-1, 3)[::73]).T
    via_rig = applyTransform(cameras[1].rig2cam, applyTransform(cameras[0].cam2rig, point))
    direct = rotation @ point + translation[:, None]
    cycle = rotation.T @ (direct-translation[:, None])
    checks['target_transform_vs_rig_max_m'] = float(np.max(np.abs(via_rig-direct)))
    checks['transform_roundtrip_max_m'] = float(np.max(np.abs(cycle-point)))
    assert np.allclose(via_rig, direct, atol=1e-12) and np.allclose(cycle, point, atol=1e-12)
    local = rays_vis @ basis
    phi, lam = np.arcsin(np.clip(local[..., 1], -1, 1)), np.arctan2(local[..., 0], local[..., 2])
    restored_local = np.stack((np.cos(phi)*np.sin(lam), np.sin(phi), np.cos(phi)*np.cos(lam)), -1)
    restored = restored_local @ basis.T
    checks['rotated_grid_direction_roundtrip_max'] = float(np.max(np.abs(restored-rays_vis)))
    assert np.allclose(restored, rays_vis, atol=1e-12)
    checks['rotation_determinants'] = dict(visual=float(np.linalg.det(OCAM_FROM_VIS)), complementary=float(np.linalg.det(basis)))
    checks['invalid_rgb_nonzero_values'] = sum(int(np.count_nonzero(item[f'{side}_rgb'][~item[side]]))
        for item in [angular, *finite] for side in ('left', 'right'))
    assert checks['invalid_rgb_nonzero_values'] == 0
    checks['left_depth_invariant_grid_max_px'] = max(float(np.max(np.abs(item['left_uv']-angular['left_uv']))) for item in finite)
    assert checks['left_depth_invariant_grid_max_px'] < 1e-5

    # 直接在原图做独立的双线性公式，检查 RGB 顺序、u/v 顺序与球面采样。
    uv = angular['left_uv'].reshape(2, -1)[:, angular['left'].ravel()][:, ::100]
    u, v = uv
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    x1, y1 = np.ceil(u).astype(int), np.ceil(v).astype(int)
    wx, wy = (u-x0)[:, None], (v-y0)[:, None]
    image = images[0].astype(float)
    manual = ((1-wx)*(1-wy)*image[y0, x0] + wx*(1-wy)*image[y0, x1] +
              (1-wx)*wy*image[y1, x0] + wx*wy*image[y1, x1])
    actual = angular['left_rgb'].reshape(-1, 3)[angular['left'].ravel()][::100]
    checks['rgb_bilinear_cv2_vs_exact_max_255'] = float(np.max(np.abs(actual-manual)))
    checks['rgb_bilinear_cv2_note'] = 'OpenCV INTER_LINEAR uses 1/32-pixel coefficient table; RGB is visualization only'
    assert checks['rgb_bilinear_cv2_vs_exact_max_255'] < 9.
    far = sphere_projection(cameras, images, rays, rotation, translation, 1e6)
    both = far['right'] & angular['right']
    checks['far_1e6m_vs_angular_right_uv_max_px'] = float(np.max(np.abs(far['right_uv'][:, both]-angular['right_uv'][:, both])))
    checks['far_1e6m_vs_angular_mask_mismatch_fraction'] = float(np.mean(far['right'] != angular['right']))
    assert checks['far_1e6m_vs_angular_right_uv_max_px'] < .01
    checks['finite_vs_angular_right_mask_mismatch_pixel_fraction'] = {
        str(item['radius_m']): float(np.mean(item['right'] != angular['right'])) for item in finite}
    # 不要求任意自定义半径一定与远场差很多：半径趋于无穷时，本来就应收敛。
    checks['status'] = 'PASS'
    return checks
