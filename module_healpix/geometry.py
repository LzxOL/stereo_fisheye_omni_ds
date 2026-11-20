"""Immutable cam4/cam3 HEALPix sampling and output support."""
from pathlib import Path

import healpy as hp
import numpy as np
import torch
from torch import nn

from dataset_stereo import feature_grid, fisheye_reference_rays, project_camera
from experiments.local_spherical_grid.camera_geometry import pair_geometry
from experiments.healpix_probe.geometry_probe import HPX_TO_OCAM, build_sweep


def pad_xy_ids(nside):
    face, row, col = np.indices((12, nside, nside), dtype=np.int64)
    return hp.xyf2pix(nside, nside - 1 - row, nside - 1 - col, face, nest=True)


def hpx_rays(ids):
    return HPX_TO_OCAM @ np.stack(hp.pix2vec(ids.shape[-1], ids.ravel(), nest=True))


def candidate_support(cameras, rays, axis, left_pixels=None, left_mask=None):
    reference, target = cameras
    rotation, translation, _ = pair_geometry(reference, target)
    pixels, visible = project_camera(reference, rays, pixels=left_pixels)
    _, visible = feature_grid(reference, pixels, visible)
    if left_mask is not None:
        visible &= left_mask
    rotated = rotation @ rays
    mask = np.empty((len(axis), rays.shape[1]), dtype=bool)
    for i, rho in enumerate(axis):
        point = (rotated + rho * translation[:, None]) / rho
        uv, right = project_camera(target, point)
        _, right = feature_grid(target, uv, right)
        mask[i] = visible & right
    return mask


def output_query(nest, rays, ready):
    # Stage 2 uses the same HEALPix interpolation, with strict original
    # support and numerical zero weights excluded from mask checks.
    nside = nest.shape[-1]
    direction = HPX_TO_OCAM.T @ rays
    direction[:, ~ready] = np.array([[1.], [0.], [0.]])
    theta, phi = hp.vec2ang(direction.T)
    ids, weight = hp.get_interp_weights(nside, theta, phi, nest=True)
    owner = hp.ang2pix(nside, theta, phi, nest=True)
    order = np.argsort(nest.ravel())
    return order[ids], np.where(weight > 1e-12, weight, 0).astype(np.float32), order[owner]


class HealpixGeometry(nn.Module):
    """Build immutable geometry from calibration, with optional Stage 1 cache audit."""
    def __init__(self, cameras, axis, nside=64, output_size=(320, 480),
                 stage1_grid: Path | None = None):
        super().__init__()
        if nside < 8 or nside & (nside - 1):
            raise ValueError("HEALPix Nside must be a power of two and at least 8")
        if tuple(int(cam.id) for cam in cameras) != (4, 3):
            raise ValueError("The calibrated stereo pair is cam4 (reference), cam3 (target)")
        if any((cam.height, cam.width) != (768, 800) for cam in cameras):
            raise ValueError("HEALPix expects the original 768×800 fish-eye calibration")
        axis = np.asarray(axis, dtype=np.float64)
        if axis.shape != (96,) or not np.isfinite(axis).all() or axis[0] <= 0 or not (np.diff(axis) > 0).all():
            raise ValueError("Expected 96 positive and increasing frozen inverse-depth candidates")
        ids = pad_xy_ids(nside)
        sweep = build_sweep(ids, cameras, axis)
        if stage1_grid is not None:
            with np.load(stage1_grid, allow_pickle=False) as checked:
                for key, data in (("nest", ids), ("inverse_depth", axis), ("left_grid", sweep.left_grid),
                                  ("right_grid", sweep.right_grid), ("left_valid", sweep.left_valid),
                                  ("stereo_valid", sweep.joint)):
                    np.testing.assert_array_equal(data, checked[key], err_msg=key)

        children = pad_xy_ids(2 * nside)
        n = nside
        expected_parents = np.repeat(np.repeat(ids, 2, axis=1), 2, axis=2)
        np.testing.assert_array_equal(children // 4, expected_parents)
        refined_mask = candidate_support(cameras, hpx_rays(children), axis).reshape(96, 12, 2*n, 2*n)
        rays, pixels, ready = fisheye_reference_rays(cameras[0], output_size)
        _, visible = project_camera(cameras[0], rays, pixels=pixels)
        ready &= visible
        direct = candidate_support(cameras, rays, axis, left_pixels=pixels, left_mask=ready)
        indices, weights, owners = output_query(children, rays, ready)

        for name, data in {
            "axis": axis.astype(np.float32), "nest": ids, "child_nest": children,
            "rays": sweep.rays.astype(np.float32), "left_grid": sweep.left_grid,
            "right_grid": sweep.right_grid, "left_valid": sweep.left_valid,
            "stereo_valid": sweep.joint, "child_candidates": refined_mask.transpose(1, 0, 2, 3).copy(),
            "output_candidates": direct.reshape(96, *output_size), "query_indices": indices,
            "query_weights": weights, "query_owner": owners, "query_ready": ready,
        }.items():
            self.register_buffer(name, torch.from_numpy(np.ascontiguousarray(data)), persistent=False)
        self.nside = nside
        self.output_size = output_size

    def project(self, indices, support):
        batch = indices.shape[0]
        values = indices.flatten(3).reshape(batch, -1)
        support = support.reshape(batch, -1)
        valid = support[:, self.query_owner] & self.query_ready
        projected = values.new_zeros((batch, self.query_owner.numel()))
        for bins, weights in zip(self.query_indices, self.query_weights):
            seen = support[:, bins]
            valid = valid & (seen | (weights == 0))
            projected = projected + values[:, bins].masked_fill(~seen, 0) * weights
        projected = projected.reshape(batch, 1, *self.output_size)
        valid = valid.reshape_as(projected)
        mask = self.output_candidates[None].expand(batch, -1, -1, -1)
        valid = valid & at_depth(mask, projected, dim=1)
        return projected.masked_fill(~valid, 0), valid


def at_depth(candidates, index, dim):
    """A continuous index needs both actual bracketing candidate samples."""
    low = index.floor().long().clamp(0, 95)
    high = index.ceil().long().clamp(0, 95)
    return candidates.gather(dim, low) & candidates.gather(dim, high)


def at_parent_depth(candidates, index):
    b, faces, _, h2, w2 = index.shape
    h, w = h2 // 2, w2 // 2
    # Query four different depths at each parent without materializing a
    # depth mask repeated over the complete 2N-resolution grid.
    arranged = index.reshape(b, faces, h, 2, w, 2).permute(0, 1, 3, 5, 2, 4)
    able = at_depth(candidates, arranged.reshape(b, faces, 4, h, w), dim=2)
    return able.reshape(b, faces, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape_as(index)
