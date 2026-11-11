"""Check native left-fisheye ray geometry with the project's real calibration."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import makeSphericalRays
from dataset_stereo import (StereoDataset, fisheye_reference_rays, project_camera,
                            feature_grid, rig_gt_to_reference)
from utils.common import Edict
from utils.geometry import applyTransform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db_root', required=True)
    parser.add_argument('--output', default='/tmp/left_fisheye_geometry.json')
    args = parser.parse_args()
    data = StereoDataset('omnithings', Edict(use_rgb=False, num_invdepth=96), args.db_root,
                         'cam4', 'cam3', augment=False, reference_grid='left_fisheye')
    h, w = data.sweep_size
    rays, pixels, valid = fisheye_reference_rays(data.reference, data.output_size)
    np.testing.assert_allclose(np.linalg.norm(rays[:, valid], axis=0), 1., atol=1e-12)
    directions = rays.reshape(3, *data.output_size)[:, ::2, ::2].reshape(3, -1)
    pixels = pixels.reshape(2, *data.output_size)[:, ::2, ::2].reshape(2, -1)
    _, visible = project_camera(data.reference, directions, pixels=pixels)
    left, visible = feature_grid(data.reference, pixels, visible)
    np.testing.assert_array_equal(data.grids[0][:, :, 0], left.reshape(h, w, 2))
    np.testing.assert_array_equal(data.grids[0][:, :, 0], data.grids[0][:, :, -1])
    for d in (0, 48, 95):
        point = directions / data.axis[d]
        np.testing.assert_allclose(np.linalg.norm(point[:, visible], axis=0), 1 / data.axis[d])
        rig = applyTransform(data.reference.cam2rig, point)
        np.testing.assert_allclose(applyTransform(data.reference.rig2cam, rig), point, rtol=1e-10, atol=1e-6)
        target = applyTransform(data.target.rig2cam, rig)
        projected, support = project_camera(data.target, target)
        right, support = feature_grid(data.target, projected, support)
        np.testing.assert_allclose(data.grids[1][:, :, d], right.reshape(h, w, 2))
        np.testing.assert_array_equal(data.candidate_valid[0, d], (visible & support).reshape(h, w))
    assert not np.array_equal(data.grids[1][:, :, 0], data.grids[1][:, :, -1])

    # A known Rig-centred 8 m sample, chosen by actual calibrated visibility.
    # This is a synthetic geometry check, separate from the real TIFF below.
    size = (64, 128)
    rig = makeSphericalRays(size, data.gt_phi) * 8
    left_points = applyTransform(data.reference.rig2cam, rig)
    target_points = applyTransform(data.target.rig2cam, rig)
    left_pixels, lv = project_camera(data.reference, left_points)
    _, rv = project_camera(data.target, target_points)
    candidates = np.flatnonzero(lv & rv)
    assert len(candidates)
    i = candidates[len(candidates) // 2]
    raw = np.zeros(size, dtype=np.float32)
    raw.flat[i] = 1 / 8
    gt, mask, distance = rig_gt_to_reference(raw, data.gt_phi, data.reference, data.target,
                                             data.axis, data.output_size, None,
                                             reference_grid='left_fisheye')
    expected_depth = np.linalg.norm(left_points[:, i])
    expected_index = (1 / expected_depth - data.axis[0]) / (data.axis[1] - data.axis[0])
    x = round(left_pixels[0, i] / (data.reference.width - 1) * (data.output_size[1] - 1))
    y = round(left_pixels[1, i] / (data.reference.height - 1) * (data.output_size[0] - 1))
    assert mask.sum() == 1 and mask[0, y, x]
    np.testing.assert_allclose(distance[0, y, x], expected_depth, rtol=1e-6)
    np.testing.assert_allclose(gt[0, y, x], expected_index, rtol=1e-6)

    real_gt = data.loadGTInvdepthIndex(data.train_idx[0])
    real_valid = real_gt >= 0
    assert real_valid.any() and np.isfinite(real_gt).all()
    assert real_gt[real_valid].min() >= 0 and real_gt[real_valid].max() <= 95
    result = dict(reference_grid=data.reference_grid, geometry=data.metadata,
                  grid_shapes=[list(g.shape) for g in data.grids],
                  candidate_mask_shape=list(data.candidate_valid.shape),
                  unit_rays=True, radial_depth=True, transform_roundtrip=True,
                  left_depth_invariant=True, target_projection=True, target_depth_dependent=True,
                  single_point_gt_pixel=[int(y), int(x)], single_point_gt_depth=float(expected_depth),
                  single_point_gt_index=float(expected_index),
                  left_valid_fraction=float(data.left_valid_mask.mean()),
                  any_match_fraction=float(data.candidate_valid.any(1).mean()),
                  candidate_valid_fraction=float(data.candidate_valid.mean()),
                  first_real_gt_shape=list(real_gt.shape), first_real_gt_valid_fraction=float(real_valid.mean()),
                  first_real_gt_index_range=[float(real_gt[real_valid].min()), float(real_gt[real_valid].max())])
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != 'geometry'}, indent=2))


if __name__ == '__main__':
    main()
