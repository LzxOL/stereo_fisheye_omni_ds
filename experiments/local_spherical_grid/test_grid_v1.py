"""Independent geometry/cache/interface tests; no CNN, training or GT changes."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
from adaptive_local_grid_v1 import (AdaptiveLocalSphericalGrid, GridGeometryError, HERE,
    load_config, calibration_hash, shape_bucket, pixel_to_normalized)
from camera_geometry import load_pair, pair_geometry
from dataset_stereo import project_camera, feature_grid, fisheye_reference_rays
from search_yang_grid import depth_axis

OPTIONS = None


class GridV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cameras, _, cfg, _ = load_pair(OPTIONS.data_root,frame=OPTIONS.frames[0])
        cls.axis = depth_axis(cfg)
        cls.config = load_config(OPTIONS.config)
        cls.selection = json.loads(OPTIONS.selection.read_text())
        cls.package = AdaptiveLocalSphericalGrid.build(*cls.cameras,cls.axis,cls.config,selection=cls.selection)

    def test_padding_and_mask_semantics(self):
        for p in [self.package.yin,self.package.yang]:
            h,w=p.original_shape
            self.assertEqual(p.right_uv.shape,(96,*p.shape,2))
            self.assertEqual(p.stereo_valid.shape,(96,*p.shape))
            self.assertTrue(all(n%8==0 for n in p.shape))
            self.assertEqual(p.padding_valid.sum(),h*w)
            self.assertFalse(p.stereo_valid[:,~p.padding_valid].any())
            self.assertTrue(np.isnan(p.rays_ref[~p.padding_valid]).all())
            self.assertTrue((p.right_grid[:,~p.padding_valid]==2).all())
            self.assertEqual(p.unpad(np.zeros((1,32,96,*p.shape))).shape,(1,32,96,h,w))
            np.testing.assert_allclose(np.linalg.norm(p.rays_ref[p.padding_valid],axis=-1),1,atol=1e-12)
        ref=self.package.reference
        np.testing.assert_array_equal(ref['left_only'],ref['left_valid']&~ref['stereo_valid'].any(0))

    def test_native_rays_are_not_resized_by_padding(self):
        from search_yang_grid import native_rays
        for p in [self.package.yin,self.package.yang]:
            raw,_=native_rays(p.bounds,p.basis,p.original_shape)
            np.testing.assert_array_equal(raw,p.rays_ref[:raw.shape[0],:raw.shape[1]])

    def test_ray_chart_roundtrip_and_cross_chart(self):
        rng=np.random.default_rng(1126)
        for src in [self.package.yin,self.package.yang]:
            rays=src.rays_ref[src.padding_valid][rng.integers(src.padding_valid.sum(),size=3000)]
            for dst in [self.package.yin,self.package.yang]:
                g,valid=dst.direction_to_grid(rays)
                back,support=dst.grid_to_direction(g)
                np.testing.assert_allclose(back[valid],rays[valid],atol=2e-14)
                self.assertTrue(support[valid].all())
        g,inside=self.package.correspondence('yin','yang')
        back,valid=self.package.yang.grid_to_direction(g)
        np.testing.assert_allclose(back[inside],self.package.yin.rays_ref[inside],atol=2e-14)
        self.assertTrue(valid[inside].all())

    def test_depth_projection_and_feature_coordinates(self):
        rotation,translation,_=pair_geometry(*self.cameras)
        for p in [self.package.yin,self.package.yang]:
            h,w=p.original_shape
            rays=p.rays_ref[:h,:w].reshape(-1,3).T
            pixels,lv=project_camera(self.cameras[0],rays)
            grid,lf=feature_grid(self.cameras[0],pixels,lv)
            np.testing.assert_array_equal(grid.reshape(h,w,2),p.left_grid[:h,:w])
            for d in [0,48,95]:
                points=rotation@(rays/self.axis[d])+translation[:,None]
                uv,rv=project_camera(self.cameras[1],points)
                rg,rf=feature_grid(self.cameras[1],uv,rv)
                np.testing.assert_array_equal(rg.reshape(h,w,2),p.right_grid[d,:h,:w])
                np.testing.assert_array_equal((lf&rf).reshape(h,w),p.stereo_valid[d,:h,:w])
                np.testing.assert_allclose(uv.T.reshape(h,w,2),p.right_uv[d,:h,:w])
            self.assertFalse(np.array_equal(p.right_grid[0],p.right_grid[-1]))

    def test_output_grid_and_forward_mapping(self):
        ref=self.package.reference
        rays,pixels,_=fisheye_reference_rays(self.cameras[0],(320,480))
        np.testing.assert_array_equal(ref['rays_ref'],rays.T.reshape(320,480,3))
        np.testing.assert_array_equal(ref['uv'],pixels.T.reshape(320,480,2))
        for p in [self.package.yin,self.package.yang]:
            grid,inside=p.direction_to_grid(ref['rays_ref'])
            back,valid=p.grid_to_direction(grid)
            np.testing.assert_allclose(back[inside],ref['rays_ref'][inside],atol=3e-14)
            self.assertTrue(valid[inside].all())
            expected=pixel_to_normalized(p.left_uv,(self.cameras[0].height,self.cameras[0].width))
            np.testing.assert_allclose(ref[p.name+'_to_output'][p.left_valid],expected[p.left_valid],atol=6e-8)
            # Actual OCam inverse polynomial is approximate: isolate its error
            # from the analytic chart roundtrip rather than claiming exact inversion.
            uv=p.left_uv[p.left_valid].T
            back=self.cameras[0].pixelToRay(uv)
            good=np.isfinite(back).all(0)
            back=back[:,good]/np.linalg.norm(back[:,good],axis=0)
            original=p.rays_ref[p.left_valid][good]
            angle=np.rad2deg(np.arctan2(np.linalg.norm(np.cross(back.T,original),axis=1),np.sum(back.T*original,axis=1)))
            self.assertLess(float(angle.max()),.01)

    def test_cache_roundtrip_and_wrong_calibration_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix=Path(tmp)/'geometry'
            self.package.save(prefix)
            p=AdaptiveLocalSphericalGrid.load(prefix,self.package.metadata['calibration_hash'])
            np.testing.assert_array_equal(p.yin.right_grid,self.package.yin.right_grid)
            np.testing.assert_array_equal(p.yang.stereo_valid,self.package.yang.stereo_valid)
            with self.assertRaisesRegex(GridGeometryError,'Calibration hash'):
                AdaptiveLocalSphericalGrid.load(prefix,'not-this-rig')
            p=AdaptiveLocalSphericalGrid.load(prefix,expected_config=self.config)
            different=copy.deepcopy(self.config)
            different['generation']['yin_lat_limit']=60 if self.config['generation']['yin_lat_limit']==50 else 50
            with self.assertRaisesRegex(GridGeometryError,'Config/profile mismatch'):
                AdaptiveLocalSphericalGrid.load(prefix,expected_config=different)
            doc=json.loads(prefix.with_suffix('.json').read_text())
            doc['arrays']['yin__left_grid']['sha256']='corrupt'
            prefix.with_suffix('.json').write_text(json.dumps(doc))
            with self.assertRaisesRegex(GridGeometryError,'Corrupted'):
                AdaptiveLocalSphericalGrid.load(prefix)

    def test_bucket_preserves_geometry_and_padding(self):
        # Two rigs can have different patch sizes; no resize, per-item native crop retained.
        from adaptive_local_grid_v1 import _pad
        small=copy.deepcopy(self.package)
        p=small.yang
        enlarged=(p.shape[0]+8,p.shape[1]+8)
        for k,v in list(vars(p).items()):
            if isinstance(v,np.ndarray) and k!='basis':
                axes=(1,2) if k in ['right_uv','target_valid','stereo_valid','right_grid'] else (0,1)
                fill=np.nan if k in ['rays_ref','left_uv','right_uv'] else (2 if k.endswith('grid') else False)
                setattr(p,k,_pad(v,enlarged,fill,axes))
        bucket=shape_bucket([self.package,small])['yang']
        self.assertEqual(bucket['left_grid'].shape,(2,*small.yang.shape,2))
        np.testing.assert_array_equal(bucket['left_grid'][0,:self.package.yang.shape[0],:self.package.yang.shape[1]],self.package.yang.left_grid)
        self.assertFalse(bucket['stereo_valid'][0,:,self.package.yang.shape[0]:].any())

    def test_rgb_frame_independence_and_deterministic_build(self):
        # Load different actual frames; only camera objects go into build.
        hashes=[]
        for frame in OPTIONS.frames:
            cams,images,_,_=load_pair(OPTIONS.data_root,frame=frame)
            hashes.append(calibration_hash(*cams))
            self.assertEqual(images[0].shape,(768,800,3))
            again=AdaptiveLocalSphericalGrid.build(*cams,self.axis,self.config,selection=self.package.metadata)
            np.testing.assert_array_equal(again.yin.rays_ref,self.package.yin.rays_ref)
            np.testing.assert_array_equal(again.yang.right_grid,self.package.yang.right_grid)
            np.testing.assert_array_equal(again.reference['stereo_valid'],self.package.reference['stereo_valid'])
        self.assertTrue(all(h==hashes[0] for h in hashes))

    def test_default_search_reproduces_frozen_geometry(self):
        p=AdaptiveLocalSphericalGrid.build(*self.cameras,self.axis,self.config)
        self.assertEqual(p.yin.bounds,self.package.yin.bounds)
        self.assertEqual(p.yang.bounds,self.package.yang.bounds)
        np.testing.assert_allclose(p.yang.basis,self.package.yang.basis,atol=1e-14)

    def test_no_common_fov_has_explicit_diagnostic(self):
        target=copy.deepcopy(self.cameras[1])
        target.invalid_mask=np.ones_like(target.invalid_mask)
        with self.assertRaisesRegex(GridGeometryError,'NO_COMMON_FOV'):
            AdaptiveLocalSphericalGrid.build(self.cameras[0],target,self.axis,self.config)

    def test_invalid_depth_and_selection_are_diagnosed(self):
        with self.assertRaises(GridGeometryError):
            AdaptiveLocalSphericalGrid.build(*self.cameras,self.axis[::-1],self.config)
        wrong=copy.deepcopy(self.selection)
        wrong['geometry']['target_from_reference'][0][3]+=1
        with self.assertRaisesRegex(GridGeometryError,'extrinsics'):
            AdaptiveLocalSphericalGrid.build(*self.cameras,self.axis,self.config,selection=wrong)
        wrong=copy.deepcopy(self.package.metadata)
        wrong['selection']['calibration_hash']='wrong-rig'
        with self.assertRaisesRegex(GridGeometryError,'calibration hash'):
            AdaptiveLocalSphericalGrid.build(*self.cameras,self.axis,self.config,selection=wrong)
        bad_config=copy.deepcopy(self.config)
        bad_config['network']['spatial_multiple']=4
        with self.assertRaisesRegex(GridGeometryError,'GEV constraints'):
            AdaptiveLocalSphericalGrid.build(*self.cameras,self.axis,bad_config)


def synthetic_pose_checks(cameras,axis,cfg):
    """Controlled synthetic OCam extrinsics, NEVER labeled real image experiments."""
    from scipy.spatial.transform import Rotation
    from utils.geometry import inverseTransform
    results=[]
    for yaw in [0,15,90]:
        left,right=copy.deepcopy(cameras)
        left.cam2rig=np.zeros((6,1));left.rig2cam=inverseTransform(left.cam2rig)
        right.cam2rig=np.r_[Rotation.from_euler('y',yaw,degrees=True).as_rotvec(),[.85,0,0]].reshape(6,1)
        right.rig2cam=inverseTransform(right.cam2rig)
        start=time.perf_counter()
        try:
            p=AdaptiveLocalSphericalGrid.build(left,right,axis,cfg)
            results.append(dict(synthetic=True,yaw_deg=yaw,status='PASS',coverage_gate=p.metadata['coverage_gate'],
                yin=p.yin.bounds,yang=p.yang.bounds,basis=p.yang.basis.tolist(),elapsed_seconds=time.perf_counter()-start))
        except GridGeometryError as e:
            results.append(dict(synthetic=True,yaw_deg=yaw,status='DIAGNOSED_INFEASIBLE',diagnostic=str(e),elapsed_seconds=time.perf_counter()-start))
    return results


def main():
    global OPTIONS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=HERE/'outputs/input_snapshot')
    p.add_argument('--config',type=Path,default=HERE/'grid_v1_config.yaml')
    p.add_argument('--selection',type=Path,help='Defaults to selection_file of the selected config')
    p.add_argument('--frames',type=int,nargs='+',default=[4887,4888,4889])
    p.add_argument('--output',type=Path,default=HERE/'outputs_v1_yin50/grid_v1_tests.json')
    p.add_argument('--synthetic',action='store_true')
    OPTIONS=p.parse_args()
    if OPTIONS.selection is None:
        OPTIONS.selection=HERE/load_config(OPTIONS.config)['selection_file']
    start=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GridV1Tests))
    record=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
        failures=[str(x[0])+': '+x[1] for x in result.failures],errors=[str(x[0])+': '+x[1] for x in result.errors],
        frames=OPTIONS.frames,elapsed_seconds=time.perf_counter()-start)
    if OPTIONS.synthetic and result.wasSuccessful():
        record['synthetic_pose_checks']=synthetic_pose_checks(GridV1Tests.cameras,GridV1Tests.axis,GridV1Tests.config)
    OPTIONS.output.parent.mkdir(parents=True,exist_ok=True)
    OPTIONS.output.write_text(json.dumps(record,indent=2)+'\n')
    if result.wasSuccessful():
        GridV1Tests.package.save(OPTIONS.output.parent/'grid_v1_geometry')
        (OPTIONS.output.parent/'grid_v1_selected.json').write_text(json.dumps(GridV1Tests.package.metadata,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':
    main()
