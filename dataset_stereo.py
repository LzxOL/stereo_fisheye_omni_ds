"""Minimal OmniDS stereo data adapter: original OCam/image tools, left-centred radial GT.

Only the tuple returned to train.py changes geometry; no second training loop or
new camera model is introduced. Geometry routines follow the project's existing
OCam and Rig transforms. Missing Rig-observed surfaces remain unsupervised.
"""
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
import torch
import yaml
from dataset import Dataset, makeSphericalRays
from utils.common import Edict
from utils.geometry import applyTransform
from utils.image import readImage, readImageFloat, rgb2gray, normalizeImage, pixelToGrid


def local_rays(size, angles):
    h, w = size
    center, span, emin, emax = angles
    if min(h, w) <= 1 or not (0 < span < 360 and -90 <= emin < emax <= 90):
        raise ValueError("Invalid local spherical grid size or angular bounds")
    az = np.deg2rad(np.linspace(center - span / 2, center + span / 2, w))
    el = np.deg2rad(np.linspace(emax, emin, h))
    az, el = np.meshgrid(az, el)
    # 参考相机 +Z 为光轴，az 朝 +X，正 elevation 朝 -Y（图像上方）。
    rays = np.stack([np.cos(el) * np.sin(az), -np.sin(el), np.cos(el) * np.cos(az)])
    return rays.reshape(3, -1)


def project_camera(camera, points):
    pixels = camera.rayToPixel(points)
    u, v = pixels
    valid = np.isfinite(points).all(0) & (np.linalg.norm(points, axis=0) > 0)
    valid &= np.isfinite(pixels).all(0) & (u >= 0) & (u <= camera.width - 1) & (v >= 0) & (v <= camera.height - 1)
    # 掩码查表时允许安全限幅索引，但限幅永远不会把视场外点变成有效点。
    safe_u = np.nan_to_num(u, nan=0, posinf=0, neginf=0).clip(0, camera.width - 1)
    safe_v = np.nan_to_num(v, nan=0, posinf=0, neginf=0).clip(0, camera.height - 1)
    x0, x1 = np.floor(safe_u).astype(int), np.ceil(safe_u).astype(int)
    y0, y1 = np.floor(safe_v).astype(int), np.ceil(safe_v).astype(int)
    mask = camera.invalid_mask
    valid &= ~(mask[y0, x0] | mask[y0, x1] | mask[y1, x0] | mask[y1, x1])
    return pixels, valid


def feature_grid(camera, pixels, valid):
    # 原图不缩放。CNN 的 stride=2 卷积中心为原图 2*j；不是将 W-1 线性缩到 Wf-1。
    hf, wf = (camera.height + 1) // 2, (camera.width + 1) // 2
    if min(hf, wf) <= 1:
        raise ValueError("Fisheye images are too small for the CNN sampling grid")
    grid = pixelToGrid(pixels / 2, (1, pixels.shape[1]), (hf, wf)).reshape(-1, 2)
    valid = valid & np.isfinite(grid).all(1) & (np.abs(grid) <= 1).all(1)
    return np.where(valid[:, None], grid, 2.0).astype(np.float32), valid


def estimate_angles(reference, target, axis):
    # 从真实标定投影得到共同视场的包围窗口；窗口内空洞仍由逐候选 mask 排除。
    size = (91, 181)
    base_angles = (0., 359., -89., 89.)
    rays = local_rays(size, base_angles)
    _, ref_valid = project_camera(reference, rays)
    common = np.zeros(rays.shape[1], dtype=bool)
    for rho in axis[[0, len(axis) // 2, -1]]:
        points = applyTransform(target.rig2cam, applyTransform(reference.cam2rig, rays / rho))
        _, target_valid = project_camera(target, points)
        common |= ref_valid & target_valid
    if not common.any():
        raise ValueError("No co-visible rays found for this calibrated camera pair")
    az = np.rad2deg(np.arctan2(rays[0, common], rays[2, common]))
    center = np.rad2deg(np.angle(np.exp(1j * np.deg2rad(az)).mean()))
    offset = (az - center + 180) % 360 - 180
    el = np.rad2deg(np.arcsin(-rays[1, common]))
    lo, hi = offset.min(), offset.max()
    return [float(center + (lo + hi) / 2), float(min(359., max(2., hi - lo + 2))),
            float(max(-89., el.min() - 1)), float(min(89., el.max() + 1))]



def zbuffer_indices(pixel_ids, distances, size):
    """每格保留最小径向距离，返回原点序号；空格为 -1，不插值填洞。"""
    result = np.full(size, -1, dtype=np.int64)
    if len(pixel_ids):
        order = np.argsort(distances, kind="stable")
        unique, first = np.unique(pixel_ids[order], return_index=True)
        result[unique] = order[first]
    return result


def cloud_visibility(camera, points, abs_tolerance=.01, rel_tolerance=.01):
    pixels, visible = project_camera(camera, points)
    distance = np.linalg.norm(points, axis=0)
    ids = np.flatnonzero(visible)
    u = np.rint(pixels[0, ids]).astype(int)
    v = np.rint(pixels[1, ids]).astype(int)
    pixel_ids = v * camera.width + u
    nearest = np.full(camera.height * camera.width, np.inf)
    np.minimum.at(nearest, pixel_ids, distance[ids])
    # 这只检查当前 Rig GT 点云已观测到的遮挡；不能恢复 Rig 中心不可见的表面。
    visible[ids] &= distance[ids] <= nearest[pixel_ids] * (1 + rel_tolerance) + abs_tolerance
    return visible


def rig_gt_to_reference(invdepth, gt_phi, reference, target, axis, size, angles,
                        check_occlusion=True, occlusion_abs=.01, occlusion_rel=.01):
    from dataset import makeSphericalRays
    if not isinstance(invdepth, np.ndarray) or invdepth.ndim != 2:
        raise ValueError("GT TIFF must contain a complete 2D inverse-depth image")
    raw_valid = np.isfinite(invdepth) & (invdepth > 0)
    rays = makeSphericalRays(invdepth.shape, gt_phi)
    rig_points = rays[:, raw_valid.ravel()] / invdepth[raw_valid].astype(np.float64)[None]
    ref_points = applyTransform(reference.rig2cam, rig_points)
    target_points = applyTransform(target.rig2cam, rig_points)
    distances = np.linalg.norm(ref_points, axis=0)
    ref_pixels, ref_visible = project_camera(reference, ref_points)
    _, ref_visible = feature_grid(reference, ref_pixels, ref_visible)
    target_pixels, target_visible = project_camera(target, target_points)
    _, target_visible = feature_grid(target, target_pixels, target_visible)
    if check_occlusion:
        ref_visible &= cloud_visibility(reference, ref_points, occlusion_abs, occlusion_rel)
        target_visible &= cloud_visibility(target, target_points, occlusion_abs, occlusion_rel)

    center, span, emin, emax = angles
    safe_distance = np.maximum(distances, np.finfo(np.float64).tiny)
    az = np.rad2deg(np.arctan2(ref_points[0], ref_points[2]))
    az = (az - center + 180) % 360 - 180
    el = np.rad2deg(np.arcsin(np.clip(-ref_points[1] / safe_distance, -1, 1)))
    h, w = size
    x = (az / span + .5) * (w - 1)
    y = (emax - el) / (emax - emin) * (h - 1)
    inside = ref_visible & np.isfinite(distances) & (distances > 0)
    inside &= (np.abs(az) <= span / 2) & (el >= emin) & (el <= emax)
    candidates = np.flatnonzero(inside)
    pixel_ids = np.rint(y[candidates]).astype(int) * w + np.rint(x[candidates]).astype(int)
    winner = zbuffer_indices(pixel_ids, distances[candidates], h * w)
    cells = np.flatnonzero(winner >= 0)
    point_ids = candidates[winner[cells]]
    index = (1 / distances[point_ids] - axis[0]) / (axis[1] - axis[0])
    good = target_visible[point_ids] & np.isfinite(index) & (index >= 0) & (index <= len(axis) - 1)
    gt = np.zeros(h * w, dtype=np.float32)
    valid = np.zeros(h * w, dtype=bool)
    depth = np.zeros(h * w, dtype=np.float32)
    gt[cells[good]] = index[good]
    valid[cells[good]] = True
    depth[cells[good]] = distances[point_ids[good]]
    return gt.reshape(1, h, w), valid.reshape(1, h, w), depth.reshape(1, h, w)


class StereoDataset(Dataset):
    """Keep the original (imgs, gt, valid, raw_imgs) tuple and image normalization.

    imgs contains two cameras. GT is rebuilt from the complete Rig TIFF at the
    final output raster; its values are working-axis indices, never multiplied by 2.
    """
    def __init__(self, dbname, db_opts, db_root, ref_cam, target_cam,
                 sweep_size=(160, 240), angles=(None, 130., -45., 45.),
                 val_count=512, seed=0, augment=True, gt_width=None, check_occlusion=True):
        if dbname != 'omnithings':
            raise ValueError('The stereo adapter currently supports OmniThings only')
        super().__init__(dbname, Edict(use_rgb=db_opts.use_rgb), load_lut=False,
                         train=True, db_root=db_root)
        self.root = Path(self.db_path)
        self.ref_index = int(str(ref_cam).lower().replace('cam', '')) - 1
        self.target_index = int(str(target_cam).lower().replace('cam', '')) - 1
        if self.ref_index == self.target_index or any(i not in range(4) for i in (self.ref_index, self.target_index)):
            raise ValueError('Specify two distinct cameras cam1..cam4')
        self.reference, self.target = self.ocams[self.ref_index], self.ocams[self.target_index]
        if any((cam.height, cam.width) != (768, 800) for cam in (self.reference, self.target)):
            raise ValueError('Stereo input/calibration must remain 768x800; no implicit resizing')
        self.sweep_size = tuple(int(n) for n in sweep_size)
        if any(n <= 0 or n % 8 for n in self.sweep_size):
            raise ValueError('Stereo sweep dimensions must be positive multiples of 8')
        self.output_size = tuple(2 * n for n in self.sweep_size)
        depth_count, full_count = int(db_opts.num_invdepth), self.num_invdepth
        if depth_count < 2 or full_count % depth_count:
            raise ValueError('Stereo D must divide the original full inverse-depth candidate count')
        self.axis = self.invdepths[::full_count // depth_count][:depth_count].copy()
        self.num_invdepth = depth_count
        self.min_invdepth, self.max_invdepth = float(self.axis[0]), float(self.axis[-1])
        self.sample_step_invdepth = float(self.axis[1] - self.axis[0])
        self.invdepths = self.axis
        inferred = estimate_angles(self.reference, self.target, self.axis) if any(v is None for v in angles) else angles
        self.angles = [float(inferred[i] if v is None else v) for i, v in enumerate(angles)]
        self.equirect_size = list(self.output_size)
        # Work-grid centres correspond to output pixels (2*y, 2*x); do not build
        # a second linspace with shifted endpoint geometry.
        h, w = self.sweep_size
        rays = local_rays(self.output_size, self.angles).reshape(3, *self.output_size)[:, ::2, ::2].reshape(3, -1)
        pixels, visible = project_camera(self.reference, rays)
        left, visible = feature_grid(self.reference, pixels, visible)
        left = left.reshape(h, w, 1, 2)
        self.left_valid_mask = visible.reshape(1, h, w)
        right = np.empty((h, w, depth_count, 2), dtype=np.float32)
        candidates = np.zeros((depth_count, h, w), dtype=bool)
        for d, rho in enumerate(self.axis):
            points = applyTransform(self.target.rig2cam,
                                    applyTransform(self.reference.cam2rig, rays / rho))
            pixels, target_visible = project_camera(self.target, points)
            grid, target_visible = feature_grid(self.target, pixels, target_visible)
            right[:, :, d] = grid.reshape(h, w, 2)
            candidates[d] = visible.reshape(h, w) & target_visible.reshape(h, w)
        self.grids = [np.broadcast_to(left, (h, w, depth_count, 2)), right]
        self.candidate_valid = candidates[None]
        if not candidates.any():
            raise ValueError('No jointly valid stereo candidates in the selected angular window')
        official_train, official_test = list(self.train_idx), list(self.test_idx)
        if not 0 <= val_count < len(official_train):
            raise ValueError('val_count must leave at least one official training frame')
        validation = set(np.random.default_rng(seed).permutation(official_train)[:val_count].tolist())
        self.train_idx = [f for f in official_train if f not in validation]
        self.test_idx = sorted(validation)
        self.opts.train_idx, self.opts.test_idx = self.train_idx, self.test_idx
        self.train_size, self.test_size = len(self.train_idx), len(self.test_idx)
        self.augment, self.gt_width, self.check_occlusion = augment, gt_width, check_occlusion
        self.metadata = dict(reference=f'cam{self.ref_index+1}', target=f'cam{self.target_index+1}',
                             camera_ids=[str(self.reference.id), str(self.target.id)],
                             calibration=yaml.safe_load((self.root / 'config.yaml').read_text()),
                             image_size=[768, 800], sweep_size=list(self.sweep_size),
                             output_size=list(self.output_size), angles=self.angles,
                             inverse_depth_values=self.axis.tolist(), full_num_invdepth=int(full_count),
                             index_units='continuous_working_candidate_0_to_D_minus_1',
                             grid_convention='output_rays_strided_2_CNN_stride2_centres',
                             invalid_mask_sha256=[hashlib.sha256(c.invalid_mask.tobytes()).hexdigest()
                                                  for c in (self.reference, self.target)],
                             gt_phi=float(self.gt_phi), gt_width=gt_width, check_occlusion=check_occlusion,
                             occlusion_scope='observed_Rig_point_cloud_only', use_rgb=self.use_rgb,
                             validation_frames=self.test_idx, split_seed=seed,
                             official_train_count=len(official_train), official_test_count=len(official_test))

    def __len__(self):
        return len(self.train_idx)

    def loadImages(self, fidx, out_raw_imgs=False, use_rgb=False):
        raw = []
        cameras = (self.reference, self.target)
        for index, camera in zip((self.ref_index, self.target_index), cameras):
            path = self.root / (self.img_fmt % (index + 1, fidx))
            image = readImage(str(path))
            if image.shape[:2] != (camera.height, camera.width):
                raise ValueError(f'Image/calibration resolution mismatch: {path}')
            raw.append(image[..., :3] if image.ndim == 3 else image)
        if self.augment and fidx in self.train_idx:
            stacked = np.asarray(self.augmentor.photo_aug(Image.fromarray(np.concatenate(raw, axis=0))))
            raw = list(np.split(stacked, 2, axis=0))
        images = []
        for image, camera in zip(raw, cameras):
            if not use_rgb and image.ndim == 3:
                image = rgb2gray(image, channel_wise_mean=True)
            image = normalizeImage(image, camera.invalid_mask)
            image = image[None] if image.ndim == 2 else image.transpose(2, 0, 1)
            if use_rgb and image.shape[0] == 1:
                image = np.repeat(image, 3, axis=0)
            if not np.isfinite(image).all():
                raise ValueError(f'Non-finite normalized image, frame {fidx}')
            images.append(image.astype(np.float32))
        return (images, raw) if out_raw_imgs else images

    def loadGTInvdepthIndex(self, fidx, remove_gt_noise=True, morph_win_size=5):
        # This overrides, rather than calls, the original cropped Rig-centred loader.
        if self.gt_width is not None:
            path = self.root / (self.gt_depth_fmt % (self.gt_width, fidx))
        else:
            paths = list(self.root.glob(f'omnidepth_gt_*/{fidx:05d}.tiff'))
            if len(paths) != 1:
                raise ValueError(f'Cannot uniquely identify full GT frame {fidx}; specify --gt_width')
            path = paths[0]
        raw = readImageFloat(str(path), read_or_die=False)
        if not isinstance(raw, np.ndarray) or raw.ndim != 2:
            raise ValueError(f'Expected a complete 2D inverse-depth TIFF: {path}')
        index, valid, _ = rig_gt_to_reference(raw, self.gt_phi, self.reference, self.target,
                                             self.axis, self.output_size, self.angles,
                                             self.check_occlusion)
        index, valid = index[0], valid[0]
        floor = np.floor(index).astype(int).clip(0, self.num_invdepth - 1)
        ceil = np.ceil(index).astype(int).clip(0, self.num_invdepth - 1)
        y, x = np.indices(self.output_size)
        work = self.candidate_valid[0]
        # Match the network's output support mask and continuous D interpolation.
        valid &= work.any(0)[y // 2, x // 2]
        valid &= work[floor, y // 2, x // 2] & work[ceil, y // 2, x // 2]
        return np.where(valid, index, -1).astype(np.float32)

    def makeVisImage(self, *args, **kwargs):
        raise NotImplementedError('The original four-view panorama visualizer does not apply to stereo')
