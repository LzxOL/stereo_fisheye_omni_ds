"""OmniThings HEALPix rays and calibrated feature-sampling grids."""
from dataclasses import dataclass
from hashlib import file_digest
import json
from pathlib import Path

import healpy as hp
import numpy as np

from dataset_stereo import feature_grid, project_camera
from experiments.local_spherical_grid.camera_geometry import pair_geometry
from utils.common import EPS
from utils.geometry import applyTransform

# HEALPix (X,Y,Z) -> camera (u right, v down, optical forward).
# lon=0,lat=0 points forward, positive lon points left, the north pole points up.
HPX_TO_OCAM = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])


@dataclass
class Sweep:
    nside: int
    nest: np.ndarray
    rays: np.ndarray
    left_uv: np.ndarray
    right_uv_selected: np.ndarray
    selected: np.ndarray
    left_grid: np.ndarray
    right_grid: np.ndarray
    left_valid: np.ndarray
    right_valid: np.ndarray
    joint: np.ndarray
    report: dict


def candidate_axis(selection_path, config):
    """Use the frozen Yin–Yang axis and independently check the Dataset formula."""
    data = json.loads(selection_path.read_text())
    frozen = np.asarray(data["inverse_depth_candidates"], dtype=np.float64)
    opts = config["config"] | config["dataset"]
    low = 1 / opts.get("max_depth", 1 / EPS)
    high = 1 / opts["omnimvs_sweep_min_depth"]
    full_count = opts.get("num_invdepth", 192)
    step = (high - low) / (full_count - 1)
    derived = np.arange(low, high + step, step, dtype=np.float64)[::full_count // 96][:96]
    np.testing.assert_array_equal(frozen, derived)
    return frozen


def stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("No finite samples")
    return {
        "n": int(values.size),
        **dict(zip(("min", "p05", "median", "p95", "p99", "max"),
                   map(float, np.quantile(values, [0, .05, .5, .95, .99, 1])))),
    }


def unit(vectors):
    return vectors / np.linalg.norm(vectors, axis=0, keepdims=True)


def angle(a, b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a.T, b.T), axis=1),
                                 np.sum(a * b, axis=0)))


def roundtrip(camera, uv, truth):
    back = camera.pixelToRay(uv)
    keep = np.isfinite(back).all(axis=0)
    if not keep.any():
        raise ValueError("No valid inverse OCam projections")
    back = unit(back[:, keep])
    errors = {
        "inverse_nonfinite": int((~keep).sum()),
        "angle_deg": stats(angle(back, truth[:, keep])),
    }
    reprojected = camera.rayToPixel(back)
    uv_error = np.linalg.norm(reprojected - uv[:, keep], axis=0)
    # Retain any model-FOV disagreement separately from the actual pixel error.
    projected = (reprojected > -1e5).all(axis=0) & np.isfinite(reprojected).all(axis=0)
    errors["forward_fov_rejected_after_inverse"] = int((~projected).sum())
    errors["pixel_error_on_reprojected"] = stats(uv_error[projected])
    back_all = np.full((3, uv.shape[1]), np.nan)
    back_all[:, keep] = back
    return errors, back_all


def feature_metric(camera, uv):
    """Local angular Jacobian of the original image lift (degrees per pixel)."""
    du, dv = np.eye(2)
    # One central difference per raw-image pixel axis. This is a local metric,
    # not a linear extrapolation across the whole CNN receptive field.
    columns = []
    for axis in (du, dv):
        offsets = axis[:, None]
        columns.append((unit(camera.pixelToRay(uv + offsets)) -
                        unit(camera.pixelToRay(uv - offsets))) / 2)
    J = np.stack(columns, axis=-1).transpose(1, 0, 2)
    supported = np.isfinite(J).all(axis=(1, 2))
    singular = np.degrees(np.linalg.svd(J[supported], compute_uv=False))
    return {
        "finite": int(supported.sum()),
        "nonfinite": int((~supported).sum()),
        "large_deg_per_pixel": stats(singular[:, 0]),
        "small_deg_per_pixel": stats(singular[:, 1]),
        "anisotropy": stats(singular[:, 0] / singular[:, 1]),
    }


def build_sweep(nest, cameras, axis):
    nside = nest.shape[-1]
    faces = (12, nside, nside)
    hpx = np.stack(hp.pix2vec(nside, nest.ravel(), nest=True))
    rays = HPX_TO_OCAM @ hpx
    np.testing.assert_allclose(np.linalg.det(HPX_TO_OCAM), 1.)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=0), 1., rtol=0, atol=1e-14)
    left, right = cameras
    rotation, translation, pair = pair_geometry(left, right)
    luv, left_image_valid = project_camera(left, rays)
    lg, left_valid = feature_grid(left, luv, left_image_valid)

    dcount, count = len(axis), rays.shape[1]
    right_grid = np.empty((dcount, count, 2), np.float32)
    right_valid = np.empty((dcount, count), bool)
    ruv_selected = []
    selected = np.array([0, dcount // 2, dcount - 1])
    details = {}
    left_rt, _ = roundtrip(left, luv[:, left_valid], rays[:, left_valid])
    for d, rho in enumerate(axis):
        # Equivalent to R @ (r/rho) + t for every positive candidate.
        # Scaling by rho before OCam yields stable directions at the far layer.
        right_ray = rotation @ rays + rho * translation[:, None]
        # OCam adds EPS to the radial norm. Apply it at the same point scale as
        # production so the normalized equivalent does not change that term.
        point = right_ray / rho
        ruv, visible = project_camera(right, point)
        right_grid[d], right_valid[d] = feature_grid(right, ruv, visible)
        if d not in selected:
            continue
        ruv_selected.append(ruv.T)
        via_rig = applyTransform(right.rig2cam, applyTransform(left.cam2rig, rays / rho))
        other_uv, other_mask = project_camera(right, via_rig)
        usable = left_valid & right_valid[d]
        np.testing.assert_array_equal(visible, other_mask)
        delta = np.linalg.norm((ruv - other_uv)[:, usable], axis=0)
        np.testing.assert_allclose(delta, 0., atol=1e-6)
        rt, back = roundtrip(right, ruv[:, usable], unit(via_rig[:, usable]))
        # Project the lifted target ray onto the epipolar plane defined by the
        # true reference ray and baseline, in target coordinates.
        ref_rotated = rotation @ rays[:, usable]
        normals = np.cross(ref_rotated.T, np.broadcast_to(translation, (ref_rotated.shape[1], 3))).T
        normals = unit(normals)
        epipolar = np.degrees(np.arcsin(np.clip(np.abs((normals * back).sum(axis=0)), 0, 1)))
        details[str(d)] = {
            "rho_m_inv": float(rho),
            "radial_distance_m": float(1 / rho),
            "composed_vs_via_rig_px": stats(delta),
            "right_roundtrip": rt,
            "epipolar_angle_deg": stats(epipolar),
            "right_angular_jacobian": feature_metric(right, ruv[:, usable]),
        }

    joint = left_valid[None] & right_valid
    for grid, mask in ((lg, left_valid), (right_grid, right_valid)):
        assert np.isfinite(grid).all()
        assert np.array_equal((np.abs(grid) <= 1).all(axis=-1), mask)
    counts = joint.sum(axis=1)
    assert counts.min() > 0 and counts.max() < count
    report = {
        "pair_geometry": pair,
        "nside": nside,
        "cells": count,
        "solid_angle_sr": float(hp.nside2pixarea(nside)),
        "hpx_to_ocam": HPX_TO_OCAM.tolist(),
        "left_image_cells": int(left_image_valid.sum()),
        "left_feature_cells": int(left_valid.sum()),
        "left_fraction": float(left_valid.mean()),
        "joint_fraction_all_candidates": float(joint.mean()),
        "joint_fraction_given_left": float(joint.sum() / (left_valid.sum() * dcount)),
        "joint_any_depth_fraction": float(joint.any(axis=0).mean()),
        "joint_all_depths_fraction": float(joint.all(axis=0).mean()),
        "depth_joint_counts": counts.tolist(),
        "depth_right_counts": right_valid.sum(axis=1).tolist(),
        "joint_middle_by_face": joint[dcount // 2].reshape(faces).sum(axis=(1, 2)).tolist(),
        "left_roundtrip": left_rt,
        "left_angular_jacobian": feature_metric(left, luv[:, left_valid]),
        "selected_depth_checks": details,
    }
    return Sweep(nside, nest, rays.T.reshape(*faces, 3), luv.T.reshape(*faces, 2),
                 np.stack(ruv_selected).reshape(-1, *faces, 2), selected,
                 lg.reshape(*faces, 2), right_grid.reshape(dcount, *faces, 2),
                 left_valid.reshape(faces), right_valid.reshape(dcount, *faces),
                 joint.reshape(dcount, *faces), report)


def save_grids(path, sweep, axis):
    """Keep invalid entries at the original out-of-bounds sentinel, with masks."""
    np.savez_compressed(path, nest=sweep.nest, rays_ocam=sweep.rays,
                        inverse_depth=axis, left_uv=sweep.left_uv,
                        right_uv_selected=sweep.right_uv_selected, selected=sweep.selected,
                        left_grid=sweep.left_grid, right_grid=sweep.right_grid,
                        left_valid=sweep.left_valid, right_valid=sweep.right_valid,
                        stereo_valid=sweep.joint)


def hash_file(path: Path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()
