"""Adaptive Local Yin–Yang V1: geometry only, no model/GT/learned fusion.

Arrays use [H,W,3] unit rays in the LEFT OCam frame; uv=(column,row) in
original image pixels. A basis has LOCAL AXES AS COLUMNS in the visual frame.
Column vectors: r_ocam=A @ Q @ r_local; row arrays: r_local=r_ocam @ A @ Q.
A=diag(-1,-1,1), Q is a proper rotation, angles are degrees.
Normalized coordinates always use align_corners=True, with no clipping.
"""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import subprocess
import sys
import os
import tempfile

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from camera_geometry import OCAM_FROM_VIS as A, pair_geometry, spherical_rays
from dataset_stereo import project_camera, feature_grid, fisheye_reference_rays

VERSION = 'adaptive-local-yinyang-v1.0'


class GridGeometryError(ValueError):
    """No complete grid can be built under the supplied constraints."""


def load_config(path=None):
    cfg = yaml.safe_load(Path(path or HERE/'grid_v1_config.yaml').read_text())
    if cfg['version'] != VERSION or cfg['conventions']['align_corners'] is not True:
        raise GridGeometryError('Unsupported geometry version / align_corners convention')
    return cfg


def array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode()+a.tobytes()).hexdigest()


def config_hash(config):
    """Same calibration can have different profiles; do not identify caches by Rig alone."""
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def camera_record(camera):
    # Include ACTUAL coefficients, both transforms and invalid_mask content, not filenames.
    keys = ('id', 'height', 'width', 'poly', 'inv_poly', 'xc', 'yc', 'c', 'd', 'e',
            'max_theta', 'cam2rig', 'rig2cam')
    record = {key: np.asarray(getattr(camera, key)).tolist() for key in keys}
    record['invalid_mask_sha256'] = array_hash(np.asarray(camera.invalid_mask, dtype=bool))
    return record


def calibration_hash(reference, target):
    text = json.dumps([camera_record(reference), camera_record(target)], sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def pad_shape(shape, multiple=8):
    return tuple((int(n)+multiple-1)//multiple*multiple for n in shape)


def native_shape(bounds, step):
    return (int(np.ceil((bounds['latitude_max_deg']-bounds['latitude_min_deg'])/step))+1,
            int(np.ceil(2*bounds['longitude_half_span_deg']/step))+1)


def pixel_to_normalized(uv, size):
    """Actual pixels -> grid_sample coordinates (no clipping, H/W explicitly supplied)."""
    h, w = size
    return np.asarray(uv)*np.array([2/(w-1), 2/(h-1)])-1


def _pad(array, shape, fill, spatial_axes):
    widths = [(0, 0)]*array.ndim
    for axis, n in zip(spatial_axes, shape):
        widths[axis] = (0, n-array.shape[axis])
    return np.pad(array, widths, constant_values=fill)


@dataclass
class SphericalPatch:
    name: str
    basis: np.ndarray
    bounds: dict
    original_shape: tuple
    rays_ref: np.ndarray
    left_uv: np.ndarray
    right_uv: np.ndarray
    left_valid: np.ndarray
    target_valid: np.ndarray
    stereo_valid: np.ndarray
    left_grid: np.ndarray
    right_grid: np.ndarray
    left_feature_valid: np.ndarray
    padding_valid: np.ndarray

    @property
    def shape(self):
        return self.left_valid.shape

    def unpad(self, tensor):
        """For [*,Hpad,Wpad] arrays/tensors (volumes use [B,C,D,H,W])."""
        h, w = self.original_shape
        return tensor[..., :h, :w]

    def direction_to_grid(self, rays_ref):
        """Return normalized [..,2] coords on the PADDED patch and analytic window mask.

        This mask alone is NOT interpolation support; use sample_patch with a
        per-depth stereo_valid mask when querying a cost volume.
        """
        rays = np.asarray(rays_ref, dtype=np.float64)
        norm = np.linalg.norm(rays, axis=-1)
        good = np.isfinite(rays).all(-1) & (norm > 0)
        local = np.nan_to_num(rays/np.where(good, norm, 1)[..., None]) @ A @ self.basis
        lon = np.rad2deg(np.arctan2(local[..., 0], local[..., 2]))
        lat = np.rad2deg(np.arcsin(np.clip(local[..., 1], -1, 1)))
        b = self.bounds
        delta = (lon-b['longitude_center_deg']+180)%360-180
        h, w = self.original_shape
        x = (delta/b['longitude_half_span_deg']+1)*(w-1)/2
        y = (b['latitude_max_deg']-lat)/(b['latitude_max_deg']-b['latitude_min_deg'])*(h-1)
        inside = good & (abs(delta) <= b['longitude_half_span_deg']+1e-8)
        inside &= (lat >= b['latitude_min_deg']-1e-8) & (lat <= b['latitude_max_deg']+1e-8)
        coords = pixel_to_normalized(np.stack((x, y), -1), self.shape)
        return np.where(inside[..., None], coords, 2.), inside

    def grid_to_direction(self, coordinates):
        """Inverse normalized PADDED coords -> unit OCam rays; padding is invalid."""
        g = np.asarray(coordinates, dtype=np.float64)
        hp, wp = self.shape
        h, w = self.original_shape
        x, y = (g[..., 0]+1)*(wp-1)/2, (g[..., 1]+1)*(hp-1)/2
        valid = np.isfinite(g).all(-1) & (x >= -1e-8) & (x <= w-1+1e-8) & (y >= -1e-8) & (y <= h-1+1e-8)
        b = self.bounds
        lon = np.deg2rad(b['longitude_center_deg']-b['longitude_half_span_deg']+x/(w-1)*2*b['longitude_half_span_deg'])
        lat = np.deg2rad(b['latitude_max_deg']-y/(h-1)*(b['latitude_max_deg']-b['latitude_min_deg']))
        rays = np.stack((np.cos(lat)*np.sin(lon), np.sin(lat), np.cos(lat)*np.cos(lon)), -1) @ self.basis.T @ A
        return np.where(valid[..., None], rays, np.nan), valid


def _make_patch(name, bounds, basis, cameras, rotation, translation, axis, cfg):
    from search_yang_grid import native_rays
    size = native_shape(bounds, cfg['generation']['native_angular_step_deg'])
    rays, _ = native_rays(bounds, basis, size)
    flat = rays.reshape(-1, 3).T
    uv, lv = project_camera(cameras[0], flat)
    lg, lf = feature_grid(cameras[0], uv, lv)
    right_uv, right_grid, target, joint = [], [], [], []
    for rho in axis:
        p = rotation @ (flat/rho)+translation[:, None]
        uv_r, rv = project_camera(cameras[1], p)
        rg, rf = feature_grid(cameras[1], uv_r, rv)
        right_uv.append(uv_r.T.reshape(*size, 2))
        right_grid.append(rg.reshape(*size, 2))
        target.append(rv.reshape(size))
        joint.append((lf & rf).reshape(size))
    padded = pad_shape(size, cfg['network']['spatial_multiple'])
    def pad(x, fill=0, axes=(0, 1)):
        return _pad(np.asarray(x), padded, fill, axes)
    return SphericalPatch(name, np.asarray(basis, dtype=np.float64), bounds, size,
        pad(rays, np.nan), pad(uv.T.reshape(*size, 2), np.nan),
        pad(right_uv, np.nan, (1, 2)), pad(lv.reshape(size), False),
        pad(target, False, (1, 2)), pad(joint, False, (1, 2)),
        pad(lg.reshape(*size, 2), 2), pad(right_grid, 2, (1, 2)),
        pad(lf.reshape(size), False), pad(np.ones(size, bool), False))


def _search(reference, target, axis, cfg):
    from search_yang_grid import depth_visibility, fit_yin, search_rotation, full_resolution_selection
    args = SimpleNamespace(**cfg['generation'])
    r, v, lo, la, _ = spherical_rays(*args.search_size)
    _, _, common = depth_visibility([reference, target], r, *pair_geometry(reference, target)[:2], axis)
    if not common.any():
        raise GridGeometryError('NO_COMMON_FOV: no valid candidate at the configured depths')
    _, yin = fit_yin(np.rad2deg(lo), np.rad2deg(la), common.any(0), args)
    extra = common.any(0) & (~yin | (abs(np.rad2deg(la)) >= args.yin_lat_limit-args.seam_deg))
    if not extra.any():
        raise GridGeometryError('YANG_NOT_NEEDED: all sampled common directions fit the Yin interior; choose a single chart or revise the explicit constraints')
    try:
        rows = search_rotation(v, extra, args)
        r, v, lo, la, area = spherical_rays(*args.analysis_size)
        _, _, common = depth_visibility([reference, target], r, *pair_geometry(reference, target)[:2], axis)
        winner, _, yin_bounds, yang_bounds, basis, *_ = full_resolution_selection(
            rows, v, common, area, np.rad2deg(lo), np.rad2deg(la), args)
    except ValueError as e:
        raise GridGeometryError('NO_FEASIBLE_YANG: '+str(e)) from e
    return dict(yin_bounds=yin_bounds, yang_bounds=yang_bounds, yang_basis_visual=basis.tolist(), selected=winner)


class AdaptiveLocalSphericalGrid:
    def __init__(self, yin, yang, reference, metadata):
        self.yin, self.yang, self.reference, self.metadata = yin, yang, reference, metadata

    @classmethod
    def build(cls, reference_camera, target_camera, inverse_depth_candidates, config=None, selection=None):
        """Build deterministically. selection=None re-searches this calibration.

        Optional selection is a validated search_result dict for this exact pair,
        not a generic preset for a new Rig. Geometry is ALWAYS recomputed.
        """
        cfg = load_config() if config is None else json.loads(json.dumps(config))
        generation=cfg['generation'];network=cfg['network']
        if network['spatial_multiple'] < 8 or network['spatial_multiple']%8 or network['depth_multiple'] < 4 or network['depth_multiple']%4:
            raise GridGeometryError('Padding multiples must preserve the actual GEV constraints: spatial 8, depth 4')
        if not (0 < generation['seam_deg'] < generation['yin_lat_limit'] < 90 and 0 < generation['yang_max_lat'] < 90):
            raise GridGeometryError('Require 0 < seam < Yin latitude <90 and 0<Yang latitude<90')
        if any(not np.isfinite(generation[k]) or generation[k]<=0 for k in
               ['native_angular_step_deg','padding_deg','boundary_step_deg','coarse_step_deg','refine_step_deg']):
            raise GridGeometryError('Angular sampling/search steps and padding must be finite positive values')
        axis = np.asarray(inverse_depth_candidates, dtype=np.float64)
        if cfg['version'] != VERSION or axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all() or (axis <= 0).any() or (np.diff(axis) <= 0).any():
            raise GridGeometryError('Invalid version / positive increasing inverse-depth candidates')
        if len(axis) != cfg['dataset']['num_depth'] or len(axis)%cfg['network']['depth_multiple']:
            raise GridGeometryError('Depth count must match config and the actual UNet depth multiple')
        if cfg['network']['cnn_stride'] != 2 or cfg['conventions']['align_corners'] is not True:
            raise GridGeometryError('V1 uses the original stride-2 CNN and align_corners=True')
        if tuple(cfg['network']['output_size']) != tuple(2*n for n in cfg['network']['sweep_size']):
            raise GridGeometryError('Legacy output_size must be 2 * sweep_size')
        rotation, translation, geometry = pair_geometry(reference_camera, target_camera)
        if selection is None:
            selection = _search(reference_camera, target_camera, axis, cfg)
        else:
            # Both the V1 metadata document and its nested selection are accepted.
            selection = json.loads(json.dumps(selection))
            if 'selection' in selection:
                selection = selection['selection']
            if 'calibration_hash' in selection:
                if selection['calibration_hash'] != calibration_hash(reference_camera,target_camera):
                    raise GridGeometryError('Selection calibration hash mismatch; re-search')
            elif 'geometry' not in selection or 'provenance' not in selection:
                raise GridGeometryError('Reused selection must identify its calibration; re-search or load a V1 cache')
            # Reject accidental reuse on a different pair, including intrinsic changes.
            if not np.allclose(selection['geometry']['target_from_reference'], geometry['target_from_reference'], atol=1e-12):
                raise GridGeometryError('Selection extrinsics do not match current calibration; re-search')
            from utils.ocam import OcamModel
            for actual, source in zip([reference_camera, target_camera], selection.get('provenance',[])):
                saved = OcamModel(); saved.setConfig(source['config']); saved.invalid_mask = actual.invalid_mask
                if camera_record(saved) != camera_record(actual):
                    raise GridGeometryError('Selection intrinsics do not match current calibration; re-search')
            if not np.array_equal(np.asarray(selection['inverse_depth_candidates']), axis):
                raise GridGeometryError('Selection uses a different depth candidate axis')
        b0, b1 = selection['yin_bounds'], selection['yang_bounds']
        q = np.asarray(selection['yang_basis_visual'], dtype=np.float64)
        if not np.allclose(q.T@q, np.eye(3), atol=1e-12) or not np.isclose(np.linalg.det(q), 1):
            raise GridGeometryError('Yang basis must be a proper orthogonal rotation')
        if not np.isclose(max(abs(b0['latitude_min_deg']), abs(b0['latitude_max_deg'])), cfg['generation']['yin_lat_limit']):
            raise GridGeometryError('Selection and Yin latitude configuration differ')
        if max(abs(b1['latitude_min_deg']), abs(b1['latitude_max_deg'])) > cfg['generation']['yang_max_lat']:
            raise GridGeometryError('Selection violates the configured Yang latitude limit')
        # Never trust a cached window solely because the camera names match.
        # Stream all D at the declared analysis resolution; no GT or RGB used.
        from search_yang_grid import window_mask, angular_coordinates
        rr, vv, ll, pp, _ = spherical_rays(*cfg['generation']['analysis_size'])
        yin_window = window_mask(np.rad2deg(ll),np.rad2deg(pp),b0)
        yl, yp = angular_coordinates(vv,q)
        yang_window = window_mask(yl,yp,b1)
        uncovered = seam_missing = 0
        _, lv_analysis = project_camera(reference_camera,rr.reshape(-1,3).T)
        for rho in axis:
            _, rv_analysis = project_camera(target_camera,rotation@(rr.reshape(-1,3).T/rho)+translation[:,None])
            common=(lv_analysis&rv_analysis).reshape(rr.shape[:2])
            uncovered += int(np.count_nonzero(common & ~(yin_window|yang_window)))
            seam = common & yin_window & (abs(np.rad2deg(pp)) >= cfg['generation']['yin_lat_limit']-cfg['generation']['seam_deg'])
            seam_missing += int(np.count_nonzero(seam & ~yang_window))
        if uncovered or seam_missing:
            raise GridGeometryError(f'INCOMPLETE_SELECTION: uncovered={uncovered}, seam_missing={seam_missing}; re-search')
        patches = [_make_patch(n, b, basis, [reference_camera, target_camera], rotation, translation, axis, cfg)
                   for n, b, basis in [('yin', b0, np.eye(3)), ('yang', b1, q)]]
        size = tuple(cfg['network']['output_size'])
        rays, pixels, good = fisheye_reference_rays(reference_camera, size)
        _, lv = project_camera(reference_camera, rays, pixels=pixels)
        lv &= good
        match = np.zeros((len(axis), *size), dtype=bool)
        for d, rho in enumerate(axis):
            uv, rv = project_camera(target_camera, rotation @ (rays/rho)+translation[:, None])
            _, rf = feature_grid(target_camera, uv, rv)
            _, lf = feature_grid(reference_camera, pixels, lv)
            match[d] = (lf & rf).reshape(size)
        ref = dict(rays_ref=rays.T.reshape(*size, 3), uv=pixels.T.reshape(*size, 2),
                   left_valid=lv.reshape(size), stereo_valid=match, left_only=lv.reshape(size)&~match.any(0))
        for patch in patches:
            coords, window = patch.direction_to_grid(ref['rays_ref'])
            ref['to_'+patch.name] = coords.astype(np.float32)
            ref['in_'+patch.name] = window & ref['left_valid']
            # Forward patch -> final output raster, via the actual OCam projection.
            coords = pixel_to_normalized(patch.left_uv, (reference_camera.height, reference_camera.width))
            ref[patch.name+'_to_output'] = np.where(patch.left_valid[..., None], coords, 2.).astype(np.float32)
        for src,dst in [(patches[0],patches[1]),(patches[1],patches[0])]:
            coords,inside=dst.direction_to_grid(src.rays_ref)
            ref[src.name+'_to_'+dst.name]=coords.astype(np.float32)
            ref[src.name+'_in_'+dst.name]=inside & src.padding_valid
        try:
            sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            sha = 'unknown'
        metadata = dict(version=VERSION, git_commit=sha, calibration_hash=calibration_hash(reference_camera, target_camera),
            config=cfg, geometry=geometry, selection=selection, inverse_depth_candidates=axis.tolist(),
            reference_camera=camera_record(reference_camera), target_camera=camera_record(target_camera),
            mask_semantics=dict(left_valid='Original OCam image/FOV/invalid-mask validity, independent of right view',
                stereo_valid='Both image and stride-2 feature-grid projection valid at the SAME 3D candidate; not occlusion',
                padding_valid='Native chart domain only; false padding has NaN rays and grid=2',
                left_only='Left valid, no geometrically valid stereo candidate in the configured axis',
                query_valid='Require all bilinear corners with nonzero weight to have valid support; unsupported returns zero'),
            environment=dict(python=sys.version.split()[0], numpy=np.__version__, pyyaml=yaml.__version__))
        metadata['source_hashes'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),HERE/'camera_geometry.py',HERE/'search_yang_grid.py',HERE/'grid_v1_config.yaml']}
        # Make newly searched selections safe to reuse even without the legacy provenance block.
        selection['calibration_hash']=metadata['calibration_hash']
        selection['geometry']=geometry
        selection['inverse_depth_candidates']=axis.tolist()
        metadata['coverage_gate'] = dict(analysis_size=cfg['generation']['analysis_size'],num_depth=len(axis),
            uncovered_depth_direction_count=uncovered,seam_uncovered_count=seam_missing)
        metadata['profile']=cfg.get('profile','yin60_legacy')
        metadata['config_hash']=config_hash(cfg)
        return cls(*patches, ref, metadata)

    def correspondence(self, source, destination):
        """Yin↔Yang mapping on identical rays; returns coords + analytic window mask."""
        src, dst = getattr(self, source), getattr(self, destination)
        coords, valid = dst.direction_to_grid(src.rays_ref)
        return coords, valid & src.padding_valid

    def save(self, prefix):
        """Two files, no pickle: geometry.npz arrays + geometry.json version/hash/metadata."""
        prefix = Path(prefix); prefix.parent.mkdir(parents=True, exist_ok=True)
        arrays = {'reference__'+k: v for k, v in self.reference.items()}
        for patch in (self.yin, self.yang):
            for key, value in vars(patch).items():
                if isinstance(value, np.ndarray):
                    arrays[patch.name+'__'+key] = value
        document = dict(metadata=self.metadata, patches={p.name:dict(bounds=p.bounds, original_shape=p.original_shape) for p in (self.yin, self.yang)},
                        arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=array_hash(v)) for k,v in arrays.items()})
        # Atomic replacement prevents readers seeing a partially written ZIP.
        # Array hashes reject any mismatched pair during replacement; load only
        # after save returns when regenerating a cache used by other workers.
        with tempfile.NamedTemporaryFile(dir=prefix.parent,suffix='.npz',delete=False) as tmp:
            name=tmp.name
            np.savez_compressed(tmp,**arrays)
        os.replace(name,prefix.with_suffix('.npz'))
        with tempfile.NamedTemporaryFile(dir=prefix.parent,mode='w',suffix='.json',delete=False) as tmp:
            name=tmp.name
            tmp.write(json.dumps(document,indent=2)+'\n')
        os.replace(name,prefix.with_suffix('.json'))

    @classmethod
    def load(cls, prefix, expected_calibration_hash=None, expected_config=None):
        prefix = Path(prefix); doc = json.loads(prefix.with_suffix('.json').read_text())
        metadata = doc['metadata']
        if metadata['version'] != VERSION:
            raise GridGeometryError('Incompatible saved grid schema')
        if expected_calibration_hash is not None and metadata['calibration_hash'] != expected_calibration_hash:
            raise GridGeometryError('Calibration hash mismatch; regenerate this cache')
        stored_hash=config_hash(metadata['config'])
        if metadata.get('config_hash',stored_hash)!=stored_hash:
            raise GridGeometryError('Saved config hash mismatch')
        if expected_config is not None and stored_hash!=config_hash(expected_config):
            raise GridGeometryError('Config/profile mismatch; regenerate or explicitly select the legacy profile')
        with np.load(prefix.with_suffix('.npz'), allow_pickle=False) as a:
            if set(a.files) != set(doc['arrays']):
                raise GridGeometryError('Geometry array manifest mismatch')
            data = {k:a[k] for k in a.files}
        for key, value in data.items():
            spec = doc['arrays'][key]
            if list(value.shape) != spec['shape'] or str(value.dtype) != spec['dtype'] or array_hash(value) != spec['sha256']:
                raise GridGeometryError('Corrupted geometry array: '+key)
        patches = []
        for name in ['yin','yang']:
            arrays = {k.split('__')[1]:v for k,v in data.items() if k.startswith(name+'__')}
            patches.append(SphericalPatch(name=name, bounds=doc['patches'][name]['bounds'],
                original_shape=tuple(doc['patches'][name]['original_shape']), **arrays))
        reference = {k.split('__')[1]:v for k,v in data.items() if k.startswith('reference__')}
        return cls(*patches, reference, metadata)


def shape_bucket(packages):
    """Batch different Rigs without resizing: each chart pads to its batch maximum."""
    result = {}
    if not packages:
        raise ValueError('Empty geometry batch')
    axes = [p.metadata['inverse_depth_candidates'] for p in packages]
    if any(a != axes[0] for a in axes):
        raise GridGeometryError('A geometry batch must share the same candidate axis')
    for name in ['yin', 'yang']:
        patches = [getattr(p,name) for p in packages]
        size = tuple(max(p.shape[i] for p in patches) for i in (0,1))
        result[name] = dict(left_grid=np.stack([_pad(p.left_grid,size,2,(0,1)) for p in patches]),
            right_grid=np.stack([_pad(p.right_grid,size,2,(1,2)) for p in patches]),
            stereo_valid=np.stack([_pad(p.stereo_valid,size,False,(1,2)) for p in patches]),
            left_valid=np.stack([_pad(p.left_valid,size,False,(0,1)) for p in patches]),
            left_feature_valid=np.stack([_pad(p.left_feature_valid,size,False,(0,1)) for p in patches]),
            padding_valid=np.stack([_pad(p.padding_valid,size,False,(0,1)) for p in patches]),
            original_shapes=[p.original_shape for p in patches], padded_shapes=[p.shape for p in patches])
    return result


def sweep_patch(features, patch, depth_chunk=8):
    """Differentiable [B,C,D,Hpad,Wpad] left/right volumes + [B,1,D,Hpad,Wpad] mask.

    Left samples once and expands over D. Right is sampled in chunks using the
    ORIGINAL spherical_sweep implementation. Only validity is applied afterwards.
    Float32 grid geometry is retained; half features are upcast for sampling on
    devices without suitable AMP grid_sample, and results cast back to input dtype.
    """
    import torch
    import torch.nn.functional as F
    from module_cnn_only.sweep import spherical_sweep
    left, right = features
    if depth_chunk < 1 or left.shape != right.shape:
        raise ValueError('Positive depth_chunk and equal left/right feature shapes required')
    device = left.device
    lg = torch.as_tensor(patch.left_grid, device=device, dtype=torch.float32)
    rg = torch.as_tensor(patch.right_grid, device=device, dtype=torch.float32)
    mask = torch.as_tensor(patch.stereo_valid,device=device)[None,None].expand(left.shape[0],-1,-1,-1,-1)
    # Explicit float32 coordinates: never quantize geometric lookup to fp16.
    with torch.autocast(device_type=device.type, enabled=False):
        sampled_left = F.grid_sample(left.float(),lg[None].expand(left.shape[0],-1,-1,-1),align_corners=True)
        sampled_left = sampled_left.to(left.dtype).unsqueeze(2).expand(-1,-1,len(rg),-1,-1)
        chunks = []
        for start in range(0,len(rg),depth_chunk):
            grid = rg[start:start+depth_chunk].permute(1,2,0,3)
            # Both inputs are right: reuse the original 2/4-camera sampler unchanged.
            out = spherical_sweep([right.float(),right.float()],[grid,grid])[0]
            chunks.append(out.permute(0,1,4,2,3).to(right.dtype))
    sampled_right = torch.cat(chunks,2)
    return sampled_left.masked_fill(~mask,0), sampled_right.masked_fill(~mask,0), mask


def sample_reference(feature, patch):
    """Static left context [B,C,Hpad,Wpad], using LEFT-ONLY feature validity.

    Do not reuse a joint-masked depth slice for full left-view context: a left
    observation can be valid even when no target depth candidate is available.
    """
    import torch
    import torch.nn.functional as F
    grid=torch.as_tensor(patch.left_grid,device=feature.device,dtype=torch.float32)
    valid=torch.as_tensor(patch.left_feature_valid,device=feature.device)[None,None]
    with torch.autocast(device_type=feature.device.type,enabled=False):
        out=F.grid_sample(feature.float(),grid[None].expand(feature.shape[0],-1,-1,-1),align_corners=True)
    return out.to(feature.dtype).masked_fill(~valid,0),valid.expand(feature.shape[0],-1,-1,-1)


def sample_patch(values, patch, rays_ref, support):
    """Masked 2D query for features or a flattened batch of depth slices.

    values=[N,C,Hpad,Wpad], support=[N,1,Hpad,Wpad]. Queries use the same
    ray for all N. No fill/renormalization: any nonzero-weight invalid corner
    makes the output zero. This keeps known native support deficits explicit.
    """
    import torch
    import torch.nn.functional as F
    coords, analytic = patch.direction_to_grid(rays_ref)
    coords = torch.as_tensor(coords,device=values.device,dtype=torch.float32)[None].expand(values.shape[0],-1,-1,-1)
    if support.shape != (values.shape[0],1,*patch.shape):
        raise ValueError('Support must correspond to every queried feature/depth slice')
    # Check active corners explicitly rather than thresholding an averaged mask.
    xy = (coords+1)*torch.tensor([patch.shape[1]-1,patch.shape[0]-1],device=values.device)/2
    x,y=xy[...,0],xy[...,1]; x0,y0=x.floor().long(),y.floor().long()
    valid=torch.as_tensor(analytic,device=values.device)[None].expand(values.shape[0],-1,-1).clone()
    for ix,wx in [(x0,1-(x-x0)),(x0+1,x-x0)]:
        for iy,wy in [(y0,1-(y-y0)),(y0+1,y-y0)]:
            active=wx*wy>0
            inside=(ix>=0)&(ix<patch.shape[1])&(iy>=0)&(iy<patch.shape[0])
            index=(iy.clamp(0,patch.shape[0]-1)*patch.shape[1]+ix.clamp(0,patch.shape[1]-1)).flatten(1)
            observed=support[:,0].flatten(1).gather(1,index).reshape_as(x)
            valid &= ~active | (inside & observed)
    with torch.autocast(device_type=values.device.type,enabled=False):
        clean=values.float().masked_fill(~support.bool(),0)
        result=F.grid_sample(clean,coords,align_corners=True)
    result=result.to(values.dtype).masked_fill(~valid[:,None],0)
    return result,valid[:,None]
