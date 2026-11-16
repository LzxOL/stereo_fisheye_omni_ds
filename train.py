# train.py
# Training script for OmniDS

from __future__ import print_function, division

import time
import csv
import json
from argparse import ArgumentParser
# CNN-ONLY: wandb is optional; import it only when logging is requested.

# Torch libs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")

try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

# Internal modules
from dataset import Dataset, MultiDataset
from utils.common import *
from utils.image import *
# CNN-ONLY: select the network after argument parsing, before importing DINO dependencies.

# Initialize
torch.backends.cudnn.benchmark = True
torch.backends.cuda.benchmark = True

parser = ArgumentParser(description='Training for OmniDS')
parser.add_argument('--cnn_only', action='store_true', help='use stereo CNN-only static context, without DINO')
parser.add_argument('--stereo', action='store_true', help='train the two-camera CNN-only baseline')
parser.add_argument('--ref_cam', choices=['cam1', 'cam2', 'cam3', 'cam4'])
parser.add_argument('--target_cam', choices=['cam1', 'cam2', 'cam3', 'cam4'])
parser.add_argument('--sweep_size', type=int, nargs=2, default=[160, 240], help='stereo working H W; output is 2H 2W')
parser.add_argument('--reference_grid', choices=['local_erp', 'left_fisheye'], default='local_erp',
                    help='stereo ray raster: local ERP angles or full left fisheye pixels')
parser.add_argument('--sweep_azimuth_center_deg', type=float, help='default: infer from actual calibrated overlap')
parser.add_argument('--sweep_azimuth_span_deg', type=float, default=130.)
parser.add_argument('--sweep_elevation_min_deg', type=float, default=-45.)
parser.add_argument('--sweep_elevation_max_deg', type=float, default=45.)
parser.add_argument('--use_group_variance', action='store_true', help='stereo ablation: restore GWC + variance')
parser.add_argument('--use_dav2_geometry', action='store_true', help='frozen DA-V2 tokens enhance CNN Context and GRU hidden initialization')
parser.add_argument('--dav2_model_repo', default='/cache/models/Depth-Anything-V2', help='official DA-V2 source checkout')
parser.add_argument('--dav2_checkpoint', default='/cache/models/depth_anything_v2_vitb.pth', help='official DA-V2 Base weights')
parser.add_argument('--dav2_input_size', type=int, default=518, help='official resize lower bound, rounded to multiples of 14')
parser.add_argument('--val_count', type=int, default=512, help='stereo held-out official-training frames')
parser.add_argument('--seed', type=int, default=0, help='stereo split and training seed')
parser.add_argument('--gt_width', type=int, help='select complete raw GT TIFF width when multiple versions exist')
parser.add_argument('--no_augment', action='store_true', help='disable stereo shared photometric augmentation')
parser.add_argument('--no_gt_occlusion_check', action='store_true', help='disable observed-point-cloud visibility checks')
parser.add_argument('--max_train_steps', type=int, help='stereo bounded smoke run; stop after this many training batches')
parser.add_argument('--max_eval_samples', type=int, help='stereo bounded validation sample count')
parser.add_argument('--use_yinyang', action='store_true', help='use the calibrated Yin/Yang frontend and existing DA-V2/GRU backbone')
parser.add_argument('--yinyang_cache', help='frozen grid_v1_geometry prefix, without .npz suffix')
parser.add_argument('--gradient_checkpointing', action='store_true', help='recompute local refiner and stateless GRU update convolutions during backward')
parser.add_argument('--no_gradient_checkpointing', dest='gradient_checkpointing', action='store_false', help='store refiner activations for faster backward when memory permits')
parser.add_argument('--gpu_memory_fraction', type=float, help='optional allocator budget on a GPU shared with other jobs')
parser.add_argument('--num_workers', type=int, default=0, help='parallel image and left-centred GT loading')
parser.add_argument('--log_every', type=int, default=20, help='stereo loss / throughput logging interval')
parser.add_argument('--amp_init_scale', type=float, default=65536., help='initial AMP gradient scaling; not a learning-rate change')

parser.add_argument('--name', default='ROmniStereo_BEV', help="name of your experiment")
parser.add_argument('--restore_ckpt', help="restore checkpoint")
parser.add_argument('--pretrain_ckpt', help="pretrained checkpoint for finetuning")

parser.add_argument('--db_root', default='/home/work/jh_code/data', type=str, help='path to dataset')
parser.add_argument('--dbname', nargs='+', default=['omnithings'], type=str,
                    choices=['omnithings', 'omnihouse', 'sunny', 'cloudy', 'sunset'], help='databases to train')

# data options
parser.add_argument('--phi_deg', type=float, default=45.0, help='phi_deg')
parser.add_argument('--num_invdepth', type=int, default=192, metavar='N', help='number of disparity')
parser.add_argument('--equirect_size', type=int, nargs='+', default=[160, 640], help="size of out ERP.")
parser.add_argument('--use_rgb', action='store_true', help='use 3-channel rgb color images as input')

# net options
parser.add_argument('--base_channel', type=int, default=32, help='base channel of the network')
parser.add_argument('--encoder_downsample_twice', action='store_true',
                    help='the feature extractor downsample the fisheye input twice instead once.')
parser.add_argument('--num_downsample', type=int, default=1, help="resolution of the disparity field (1/2^K)")

# ERP / cross-attention options
parser.add_argument('--num_heads', type=int, default=4, help="number of attention heads")
parser.add_argument('--num_points', type=int, default=4, help="number of sampling points per head")
parser.add_argument('--num_cross_attn_layers', type=int, default=1, help="number of cross-attention layers")
# Similarity context options
parser.add_argument('--num_depth_samples', type=int, default=96, help="number of depth samples for similarity profile")
parser.add_argument('--sim_radius', type=int, default=4, help="lookup radius for similarity context")
parser.add_argument('--sim_levels', type=int, default=4, help="number of pyramid levels for similarity lookup")
parser.add_argument('--similarity_type', type=str, default='correlation',
                    choices=['variance', 'correlation', 'pairwise', 'adaptive'], help="similarity computation method")
# Spatial aggregation options for similarity profile
parser.add_argument('--use_spatial_aggregation', action='store_true', default=True,
                    help='apply spatial aggregation on similarity profile (recommended)')
parser.add_argument('--spatial_agg_type', type=str, default='conv',
                    choices=['conv', 'deformable', 'multiscale'],
                    help='type of spatial aggregation: conv (simple), deformable (instance-aware), multiscale (ASPP)')
parser.add_argument('--spatial_hidden_dim', type=int, default=32,
                    help='hidden dimension for spatial aggregation conv')
parser.add_argument('--spatial_num_layers', type=int, default=2,
                    help='number of conv layers for spatial aggregation')

# DINO options
parser.add_argument('--use_dino_context', action='store_true',
                    help='deprecated no-op; DINO context is always used')
parser.add_argument('--dino_model', type=str, default='dinov3_vits16',
                    choices=['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'],
                    help="DINOv3 model variant (patch 16)")
parser.add_argument('--freeze_dino', action='store_true', default=True,
                    help='freeze DINO backbone weights')
parser.add_argument('--context_fusion_type', type=str, default='pointwise',
                    choices=['pointwise', 'gated', 'local', 'attention'],
                    help='DINO-CNN fusion type: pointwise (fast), gated, local, attention (slow)')
parser.add_argument('--no_dino_cross_attn', action='store_true',
                    help='ablation: use fixed grid sampling instead of deformable cross-attention for DINO→ERP')
parser.add_argument('--use_dino_sim', action='store_true',
                    help='use DINO features (instead of CNN) for similarity volume construction')

# GEV options (separate GEV with 3D UNet)
parser.add_argument('--use_gev', action='store_true',
                    help='deprecated no-op; the GEV branch is always used')
parser.add_argument('--gev_num_groups', type=int, default=8,
                    help='number of groups for group-wise correlation')
parser.add_argument('--gev_reg_channels', type=int, nargs='+', default=[16, 32, 48],
                    help='channels for 3D UNet regularization (3 stages)')
parser.add_argument('--gev_num_pyramid_levels', type=int, default=2,
                    help='number of depth-axis pyramid levels for GEV lookup')
parser.add_argument('--gev_radius', type=int, default=4,
                    help='lookup radius for GEV depth-axis sampling')
parser.add_argument('--gev_use_spatial_downsample', action='store_true', default=True,
                    help='spatial 2x downsample before 3D UNet')

parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--fix_bn', action='store_true', help='fix batch normalization')
parser.add_argument('--distilled', action='store_true',
                    help='build and use the distilled student encoder')
parser.add_argument('--triton', action='store_true',
                    help='use Triton fused kernels for GEV/similarity')

# training options
parser.add_argument('--total_epochs', type=int, default=100, help='total epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--train_iters', type=int, default=12,
                    help="number of updates to the disparity field in each forward pass.")
parser.add_argument('--lr', type=float, default=0.0005, help="max learning rate.")
parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")
parser.add_argument('--wandb', action='store_true', help='log training to wandb')
    
args = parser.parse_args()
if args.restore_ckpt and args.pretrain_ckpt:
    parser.error('Use --restore_ckpt to resume OR --pretrain_ckpt to initialize a new run')

if args.stereo:
    args.cnn_only = True
    if args.dbname != ['omnithings'] or not args.ref_cam or not args.target_cam or args.ref_cam == args.target_cam:
        parser.error('--stereo requires OmniThings and two distinct --ref_cam/--target_cam')
    if args.distilled or args.use_dino_sim or args.triton or args.encoder_downsample_twice or args.num_downsample != 1:
        parser.error('Stereo baseline requires the original stride-2 CNN; DINO, student and Triton are unsupported')
    explicit_depth = any(s == '--num_invdepth' or s.startswith('--num_invdepth=') for s in sys.argv[1:])
    if not explicit_depth:
        args.num_invdepth = args.num_depth_samples
    if args.num_invdepth != args.num_depth_samples:
        parser.error('Stereo uses one working candidate axis: --num_invdepth must equal --num_depth_samples')
    output_size = [2 * n for n in args.sweep_size]
    explicit_size = any(s == '--equirect_size' or s.startswith('--equirect_size=') for s in sys.argv[1:])
    if explicit_size and args.equirect_size != output_size:
        parser.error('Stereo --equirect_size must equal 2 * --sweep_size')
    args.equirect_size = output_size
    if args.val_count < 0 or any(n is not None and n < 1 for n in (args.max_train_steps, args.max_eval_samples)):
        parser.error('Smoke limits must be positive, val_count nonnegative')
    if args.reference_grid == 'left_fisheye' and args.sweep_azimuth_center_deg is not None:
        parser.error('A left-fisheye pixel grid has no ERP azimuth centre')
    if args.use_dav2_geometry and (args.reference_grid != 'left_fisheye' or args.base_channel != 32):
        parser.error('--use_dav2_geometry requires --reference_grid left_fisheye and base_channel=32')
    if args.name == parser.get_default('name'):
        args.name = 'OmniDS_Stereo_DAV2_Context' if args.use_dav2_geometry else 'OmniDS_Stereo_Basic'
elif (args.ref_cam or args.target_cam or args.max_train_steps is not None or args.max_eval_samples is not None
      or args.reference_grid != 'local_erp'):
    parser.error('Camera-pair and bounded-smoke options require --stereo')
if args.use_dav2_geometry:
    if not args.stereo or args.dav2_input_size < 14:
        parser.error('--use_dav2_geometry requires --stereo and dav2_input_size>=14')
    from module_cnn_only.dav2_geometry import prepare_dav2_rgb
if args.use_yinyang and (not args.stereo or args.reference_grid != 'left_fisheye' or not args.yinyang_cache):
    parser.error('--use_yinyang requires --stereo --reference_grid left_fisheye --yinyang_cache PREFIX')

if args.cnn_only and not args.stereo:
    parser.error('--cnn_only uses stereo static context; add --stereo and camera pair, or use original four-camera OmniDS without --cnn_only')

if args.cnn_only:
    if args.distilled or args.use_dino_sim:
        parser.error('--cnn_only cannot be combined with --distilled or --use_dino_sim')
    from module_cnn_only.network import OmniDS
    from module_cnn_only.loss_functions import sequence_loss
    if args.name == parser.get_default('name'):
        args.name = 'OmniDS_CNN_only'
else:
    from module.network import OmniDS
    from module.loss_functions import sequence_loss
if args.wandb:
    import wandb

opts = Edict()
# Dataset & sweep arguments
opts.name = args.name
opts.model_dir = os.path.join('./checkpoints', args.name)

opts.snapshot_path = args.restore_ckpt
opts.pretrain_path = args.pretrain_ckpt
opts.dbname = args.dbname
opts.db_root = args.db_root

opts.data_opts = Edict()
opts.data_opts.phi_deg = args.phi_deg
opts.data_opts.num_invdepth = args.num_invdepth
opts.data_opts.equirect_size = args.equirect_size
opts.data_opts.num_downsample = args.num_downsample
opts.data_opts.use_rgb = args.use_rgb

opts.net_opts = Edict()
opts.net_opts.cnn_only = args.cnn_only
opts.net_opts.base_channel = args.base_channel
opts.net_opts.num_invdepth = opts.data_opts.num_invdepth
opts.net_opts.use_rgb = opts.data_opts.use_rgb
opts.net_opts.encoder_downsample_twice = args.encoder_downsample_twice
opts.net_opts.num_downsample = args.num_downsample
opts.net_opts.mixed_precision = args.mixed_precision
opts.net_opts.fix_bn = args.fix_bn
if args.stereo:
    opts.net_opts.num_cams = 2
    opts.net_opts.use_group_variance = args.use_group_variance
    opts.net_opts.reference_grid = args.reference_grid
    opts.net_opts.use_dav2_geometry = args.use_dav2_geometry
    opts.net_opts.use_yinyang = args.use_yinyang
    opts.net_opts.training_geometry = args.use_yinyang
    opts.net_opts.yinyang_output = 'left_fisheye' if args.use_yinyang else 'native'
    opts.net_opts.gradient_checkpointing = args.gradient_checkpointing
    if args.use_dav2_geometry:
        opts.net_opts.dav2_fusion_target = 'context'
        opts.net_opts.dav2_model_repo = args.dav2_model_repo
        opts.net_opts.dav2_checkpoint = args.dav2_checkpoint
        opts.net_opts.dav2_input_size = args.dav2_input_size
# ERP / cross-attention
opts.net_opts.equi_h = args.equirect_size[0]
opts.net_opts.equi_w = args.equirect_size[1]
opts.net_opts.phi_deg = args.phi_deg
opts.net_opts.num_heads = args.num_heads
opts.net_opts.num_points = args.num_points
opts.net_opts.num_cross_attn_layers = args.num_cross_attn_layers
# Similarity context options
opts.net_opts.num_depth_samples = args.num_depth_samples
opts.net_opts.sim_radius = args.sim_radius
opts.net_opts.sim_levels = args.sim_levels
opts.net_opts.similarity_type = args.similarity_type
# Spatial aggregation options
opts.net_opts.use_spatial_aggregation = args.use_spatial_aggregation
opts.net_opts.spatial_agg_type = args.spatial_agg_type
opts.net_opts.spatial_hidden_dim = args.spatial_hidden_dim
opts.net_opts.spatial_num_layers = args.spatial_num_layers

# DINO context options
opts.net_opts.dino_model = args.dino_model
opts.net_opts.freeze_dino = args.freeze_dino
opts.net_opts.context_fusion_type = args.context_fusion_type
opts.net_opts.no_dino_cross_attn = args.no_dino_cross_attn
opts.net_opts.use_dino_sim = args.use_dino_sim

# GEV options (v5: separate GEV with 3D UNet)
opts.net_opts.gev_num_groups = args.gev_num_groups
opts.net_opts.gev_reg_channels = tuple(args.gev_reg_channels)
opts.net_opts.gev_num_pyramid_levels = args.gev_num_pyramid_levels
opts.net_opts.gev_radius = args.gev_radius
opts.net_opts.gev_use_spatial_downsample = args.gev_use_spatial_downsample

# Distillation
opts.distilled = args.distilled
opts.net_opts.use_student = args.distilled
opts.net_opts.use_triton = args.triton

opts.total_epochs = args.total_epochs
opts.batch_size = args.batch_size
opts.train_iters = args.train_iters
opts.lr = args.lr
opts.wdecay = args.wdecay


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(model, num_steps):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=opts.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, opts.lr, num_steps+100,
                                              pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


def train(epoch_total, load_state):
    if args.gpu_memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    if args.use_dav2_geometry and not opts.snapshot_path and osp.isdir(opts.model_dir):
        if any(name.endswith('.pth') for name in os.listdir(opts.model_dir)):
            raise ValueError('DA-V2 run already has checkpoints; choose a new --name or explicitly restore')
    if args.stereo:
        from dataset_stereo import StereoDataset
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        data = StereoDataset(opts.dbname[0], opts.data_opts, opts.db_root, args.ref_cam, args.target_cam,
                             sweep_size=args.sweep_size,
                             angles=[args.sweep_azimuth_center_deg, args.sweep_azimuth_span_deg,
                                     args.sweep_elevation_min_deg, args.sweep_elevation_max_deg],
                             val_count=args.val_count, seed=args.seed, augment=not args.no_augment,
                             gt_width=args.gt_width, check_occlusion=not args.no_gt_occlusion_check,
                             reference_grid=args.reference_grid)
        LOG_INFO('Stereo: %s, left-centred radial GT, working indices [0,%d], output %s; official test split reserved' %
                 (data.reference_grid, args.num_invdepth - 1, data.output_size))
    elif len(opts.dbname) > 1:
        data = MultiDataset(opts.dbname, opts.data_opts, db_root=opts.db_root)
    else:
        data = Dataset(opts.dbname[0], opts.data_opts, db_root=opts.db_root)
    package = None
    if args.use_yinyang:
        from experiments.local_spherical_grid.adaptive_local_grid_v1 import AdaptiveLocalSphericalGrid, calibration_hash, load_config
        package = AdaptiveLocalSphericalGrid.load(args.yinyang_cache,
                    expected_calibration_hash=calibration_hash(data.reference, data.target), expected_config=load_config())
        if not np.array_equal(package.metadata['inverse_depth_candidates'], data.axis):
            raise ValueError('Frozen grid and dataset must use exactly the same inverse-depth candidates')
        if tuple(package.reference['left_valid'].shape) != tuple(data.output_size):
            raise ValueError('Frozen grid reference output size differs from the left-fisheye GT grid')
        data.output_candidate_valid = package.reference['stereo_valid']
        data.metadata['yinyang'] = dict(geometry=package.metadata,
                                      chart_shapes={n: list(getattr(package, n).shape) for n in ('yin', 'yang')},
                                      output_projection='ray_correspondence_strict_bilinear_Q_weighted_overlap')
        LOG_INFO('Yin/Yang native charts: %s; predictions projected by rays onto left GT %s' %
                 (data.metadata['yinyang']['chart_shapes'], data.output_size))
    dbloader = torch.utils.data.DataLoader(data, batch_size=opts.batch_size,
                                           pin_memory=True, shuffle=True,
                                           num_workers=args.num_workers, drop_last=True,
                                           persistent_workers=args.num_workers > 0,
                                           multiprocessing_context='spawn' if args.num_workers else None)
    total_num_steps = len(dbloader)*opts.total_epochs

    LOG_INFO("Using stereo CNN baseline: static context, group_variance=%s" %
             args.use_group_variance if args.stereo else
             "Using module_cnn_only.network.OmniDS (static CNN context + GEV)" if args.cnn_only
             else "Using OmniDS (DINO context + separate GEV)")
    if not args.cnn_only:
        LOG_INFO(f"  DINO model: {opts.net_opts.dino_model}, freeze: {opts.net_opts.freeze_dino}")
    LOG_INFO(f"  GEV groups: {opts.net_opts.gev_num_groups}, reg_channels: {opts.net_opts.gev_reg_channels}")
    LOG_INFO(f"  GEV pyramid: {opts.net_opts.gev_num_pyramid_levels} levels, radius {opts.net_opts.gev_radius}")
    model = OmniDS(opts.net_opts, grid_package=package) if args.use_yinyang else OmniDS(opts.net_opts)
    net = nn.DataParallel(model).cuda()
    if args.use_dav2_geometry:
        LOG_INFO('DA-V2: frozen ViT-B blocks [2,5,8,11], trainable Context fusion before context/state heads; GEV unchanged')
    if opts.net_opts.fix_bn:
        net.module.freeze_bn()
    LOG_INFO("Parameter Count: %d" % count_parameters(net))

    optimizer, scheduler = fetch_optimizer(net, total_num_steps)
    scaler = GradScaler(enabled=opts.net_opts.mixed_precision, init_scale=args.amp_init_scale)

    start_epoch = 0
    optimizer_updates = 0
    resume_step = None
    if args.wandb:
        wandb.init(
            entity="entity",   
            project="Videodepth",                
            settings=wandb.Settings(start_method="fork")
        )

    if load_state:
        if opts.snapshot_path and osp.exists(opts.snapshot_path):
            snapshot = (torch.load(opts.snapshot_path, weights_only=False, map_location='cpu') if args.cnn_only
                        else torch.load(opts.snapshot_path))
            if args.cnn_only and not snapshot.get('net_opts', {}).get('cnn_only', False):
                raise ValueError('CNN-only requires a CNN-only checkpoint; original DINO checkpoints are incompatible')
            if args.stereo and snapshot.get('stereo_geometry') != data.metadata:
                raise ValueError('Restore checkpoint stereo geometry differs; do not mix camera pairs/candidate axes')
            if args.stereo:
                for key in ('use_group_variance', 'use_yinyang'):
                    if snapshot['net_opts'].get(key, False) != opts.net_opts[key]:
                        raise ValueError(f'Restore checkpoint {key} differs; pass the matching ablation switch')
                if snapshot['net_opts'].get('use_dav2_geometry', False) != args.use_dav2_geometry:
                    raise ValueError('Restore checkpoint DA-V2 geometry mode differs; weights load strictly')
                if args.use_dav2_geometry and snapshot['net_opts'].get('dav2_fusion_target', 'guide') != 'context':
                    raise ValueError('Previous DA-V2 GEV Guide checkpoint is not a Context-fusion checkpoint')
                if args.use_dav2_geometry and snapshot['net_opts'].get('dav2_input_size', 518) != args.dav2_input_size:
                    raise ValueError('Restore checkpoint DA-V2 preprocessing size differs')
            if 'net_state_dict' in snapshot.keys():
                net.load_state_dict(snapshot['net_state_dict'])
                LOG_INFO('checkpoint %s is loaded' % (opts.snapshot_path))
            if 'epoch' in snapshot.keys():
                start_epoch = snapshot['epoch'] + 1
            if 'epoch_loss' in snapshot.keys():
                epoch_loss = snapshot['epoch_loss']
            if 'optimizer' in snapshot.keys():
                optimizer.load_state_dict(snapshot['optimizer'])
            if args.stereo:
                if 'scheduler' in snapshot:
                    scheduler.load_state_dict(snapshot['scheduler'])
                if 'scaler' in snapshot:
                    scaler.load_state_dict(snapshot['scaler'])
                optimizer_updates = snapshot.get('optimizer_updates', 0)
                resume_step = snapshot.get('global_step')
            if 'epoch' in snapshot.keys() and 'epoch_loss' in snapshot.keys():
                LOG_INFO('startepoch:%d epoch_loss:%f' % (start_epoch, epoch_loss))
        elif opts.pretrain_path is None:
            sys.exit('%s do not exsits' % (opts.snapshot_path))

        if opts.pretrain_path and osp.exists(opts.pretrain_path):
            snapshot = (torch.load(opts.pretrain_path, weights_only=False, map_location='cpu') if args.cnn_only
                        else torch.load(opts.pretrain_path))
            if args.cnn_only and not snapshot.get('net_opts', {}).get('cnn_only', False):
                raise ValueError('CNN-only requires a CNN-only checkpoint; original DINO checkpoints are incompatible')
            if args.stereo and snapshot.get('net_opts', {}).get('num_cams') != 2:
                raise ValueError('Stereo pretraining requires a two-camera checkpoint; weights load strictly')
            if args.stereo and snapshot['net_opts'].get('use_dav2_geometry', False) != args.use_dav2_geometry:
                raise ValueError('Pretrain checkpoint DA-V2 mode differs; partial/silent weight loading is not supported')
            if args.use_dav2_geometry and snapshot['net_opts'].get('dav2_fusion_target', 'guide') != 'context':
                raise ValueError('Pretrain checkpoint uses previous DA-V2 Guide fusion, not Context fusion')
            if 'net_state_dict' in snapshot.keys():
                if args.cnn_only:
                    transfer = net.module.load_baseline_weights(snapshot['net_state_dict'])
                    LOG_INFO(f'Explicit baseline weight transfer: {transfer}')
                else:
                    net.load_state_dict(snapshot['net_state_dict'])
                LOG_INFO('checkpoint %s is loaded' % (opts.pretrain_path))
        elif opts.snapshot_path is None:
            sys.exit('%s do not exsits' % (opts.snapshot_path))

    grids = None if args.use_yinyang else [torch.tensor(grid, requires_grad=False).cuda() for grid in data.grids]
    candidate_mask = left_mask = None
    if args.stereo and not args.use_yinyang:
        candidate_mask = torch.tensor(data.candidate_valid, dtype=torch.bool).cuda()[None]
        left_mask = torch.tensor(data.left_valid_mask, dtype=torch.bool).cuda()[None]
    if not osp.exists(opts.model_dir):
        os.makedirs(opts.model_dir, exist_ok=True)
        LOG_INFO('"%s" directory created' % (opts.model_dir))
    total_iters = len(data)*start_epoch//opts.batch_size
    if resume_step is not None:
        total_iters = resume_step
    stop_training = False
    if args.stereo:
        with open(osp.join(opts.model_dir, 'run_config.json'), 'w') as handle:
            json.dump(dict(arguments=vars(args), geometry=data.metadata, net_opts=dict(net.module.opts)), handle, indent=2)
        step_fields = ['epoch', 'step', 'loss', 'mean_loss', 'lr', 'optimizer_updates', 'amp_scale',
                       'batch_size', 'step_seconds', 'valid_ratio', 'final_index_mae', 'peak_allocated_mib', 'peak_reserved_mib']
        metrics_path = osp.join(opts.model_dir, 'steps.csv')
        if not osp.exists(metrics_path):
            with open(metrics_path, 'w', newline='') as handle:
                csv.DictWriter(handle, fieldnames=step_fields).writeheader()

    def stereo_forward(images, raw, training):
        # 复用同一入口；双网格内部扫描和跨网格融合后仍走已训练的DA-V2 Context / 原GRU。
        rgb = (prepare_dav2_rgb(raw, args.dav2_model_repo, args.dav2_input_size).cuda()
               if args.use_dav2_geometry else None)
        batch = images[0].shape[0]
        return net.module(images, grids, iters=opts.train_iters, test_mode=not training, dav2_rgb=rgb,
                          valid_mask=None if args.use_yinyang else candidate_mask.expand(batch, -1, -1, -1, -1),
                          left_valid_mask=None if args.use_yinyang else left_mask.expand(batch, -1, -1, -1))

    for epoch in range(start_epoch, epoch_total):
        epoch_start = time.time()
        if args.stereo:
            torch.cuda.reset_peak_memory_stats()
        # training
        net.train()
        train_loss = 0
        epoch_loss = 0
        usable_batches = 0
        LOG_INFO('\nEpoch: %d' % epoch)
        if args.wandb:
            wandb.log({"Epoch": epoch}, step=total_iters)

        for step, data_blob in enumerate(dbloader):
            start_time = time.time()
            imgs, gt, valid, raw_imgs = data_blob

            imgs = [img.cuda() for img in imgs]
            valid = valid.cuda()
            gt = gt.cuda()
            if args.stereo and not valid.any():
                if args.max_train_steps:
                    raise ValueError('Smoke batch has no valid left-centred supervision')
                LOG_INFO('Skipping stereo batch without valid supervision')
                continue

            optimizer.zero_grad()
            # Pass ocams for principal point bias (if available)
            ocams = getattr(data, 'ocams', None)
            if args.stereo:
                predictions = stereo_forward(imgs, raw_imgs[0], True)
                if args.use_yinyang:
                    valid = valid.bool() & net.module.output_valid[:, 0]
                    if not valid.any():
                        raise ValueError('No overlap between projected dual-chart predictions and left GT')
            elif args.cnn_only:
                predictions = net.module.forward(imgs, grids, iters=opts.train_iters, ocams=ocams)
            else:
                predictions = net.module.forward(imgs, grids, ocams=ocams)

            if args.use_yinyang and step == 0:
                LOG_INFO('Actual tensors: %s; left prediction %s' %
                         (net.module.debug_outputs, list(predictions[-1].shape)))

            loss = sequence_loss(predictions, gt.unsqueeze(1), valid.unsqueeze(1))
            
            train_loss += loss.data
            usable_batches += 1
            # train_loss += loss2.data *0.5
            # total_loss = loss + 0.5 * loss2

            epoch_loss = train_loss / (usable_batches if args.stereo else step + 1)

            if step % 200 == 0 and not args.stereo:
                LOG_INFO("Iter %d/%d training loss = %.3f, average training loss for every step = %.3f, time = %.2f" % (total_iters - epoch*len(data),len(data), loss, epoch_loss, time.time() - start_time))
            # writer.add_scalar("train/loss", loss, total_iters)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(optimizer)
            if args.stereo:
                previous_scale = scaler.get_scale()
                scaler.update()
                # Preserve the original optimizer/scheduler, but do not advance LR
                # when AMP has skipped a stereo optimizer update.
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                    optimizer_updates += 1
            else:
                scheduler.step()
                scaler.update()

            total_iters += 1
            if args.stereo:
                # .item同步GPU后才计时，包含前向、反向和优化器更新；CSV可实时监控。
                loss_value = loss.item()
                index_mae = (predictions[-1][:, 0] - gt).abs()[valid.bool()].mean().item()
                record = dict(epoch=epoch + 1, step=total_iters, loss=loss_value, mean_loss=float(epoch_loss),
                              lr=scheduler.get_last_lr()[0], optimizer_updates=optimizer_updates,
                              amp_scale=scaler.get_scale(), batch_size=imgs[0].shape[0],
                              step_seconds=time.time() - start_time, valid_ratio=valid.float().mean().item(),
                              final_index_mae=index_mae, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                              peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
                with open(metrics_path, 'a', newline='') as handle:
                    csv.DictWriter(handle, fieldnames=step_fields).writerow(record)
                if step % args.log_every == 0:
                    LOG_INFO('Epoch %d/%d step %d/%d loss %.4f mean %.4f index-MAE %.3f valid %.1f%% %.2fs/batch; peak %.0f MiB; updates %d scale %.0f' %
                             (epoch + 1, epoch_total, step + 1, len(dbloader), loss_value, epoch_loss, index_mae,
                              100*record['valid_ratio'], record['step_seconds'], record['peak_allocated_mib'],
                              optimizer_updates, scaler.get_scale()))

            if args.stereo and args.wandb and total_iters % 300 == 0:
                wandb.log({'LR': scheduler.get_last_lr()[0], 'Train/total_loss': loss.item()}, step=total_iters)
            if args.wandb and total_iters % 300 == 0 and not args.stereo:
                wandb.log({
                    "LR": scheduler.get_last_lr()[0],
                    "Train/total_loss": loss.item(),
                }, step=total_iters)
                
                # === VIS inputs: 반드시 detach + cpu ===
                raw_imgs_vis = [img[0].detach().cpu() for img in raw_imgs]  # [768,800,3] x4
                pred_vis = data.indexToInvdepth(predictions[-1][0, 0]).detach().cpu()            # [160,640] (마지막 prediction 권장)

                if torch.is_tensor(gt):
                    gt_vis = data.indexToInvdepth(gt[0]).detach().cpu() if gt.dim() == 3 else data.indexToInvdepth(gt).detach().cpu()
                else:
                    gt_vis = data.indexToInvdepth(gt[0]).detach().cpu()  # gt가 list라면

                vis_results = data.makeVisImage(raw_imgs_vis, pred_vis, gt=gt_vis, return_all=True)
                wimages = {"Train/Predictions": []}
                input_rgb, pred_rgb, gt_rgb, err_rgb = vis_results # unpack each tuple/list

                images = {
                    f"Sample /Input": input_rgb,
                    f"Sample /Prediction": pred_rgb,
                    f"Sample /GT": gt_rgb,
                    f"Sample /Error": err_rgb
                }
                wimages["Train/Predictions"].extend(
                    [wandb.Image(img, caption=caption) for caption, img in images.items()]
                )
                wandb.log(wimages, step=total_iters)
            if args.stereo and args.max_train_steps and step + 1 >= args.max_train_steps:
                stop_training = True
                break
    
        if args.stereo and not usable_batches:
            raise ValueError('No valid stereo supervision in the entire training epoch')

        # evaluation
        net.eval()
        eval_list = data.opts.test_idx
        if args.stereo and args.max_eval_samples:
            eval_list = eval_list[:args.max_eval_samples]

        rank0 = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)
        net_eval = net.module if hasattr(net, "module") else net  # DDP면 원본 모듈로 eval

        if rank0:
            errors = np.zeros((len(eval_list), 5))
            for d in range(len(eval_list)):
                fidx = eval_list[d]
                imgs, gt, valid, raw_imgs = data.loadSample(fidx)
                if args.stereo and not valid.any():
                    errors[d, :] = np.nan
                    continue
                imgs = [torch.Tensor(img).unsqueeze(0).cuda() for img in imgs]
                # breakpoint()
                # Pass ocams for LoRA adaptation (same as training)
                ocams = getattr(data, 'ocams', None)
                with torch.no_grad():
                    if args.stereo:
                        invdepth_idx = stereo_forward(imgs, raw_imgs[0][None], False)
                        if args.use_yinyang:
                            valid = valid.astype(bool) & toNumpy(net_eval.output_valid[0, 0])
                    elif args.cnn_only:
                        invdepth_idx = net_eval(imgs, grids, test_mode=True, iters=opts.train_iters)
                    else:
                        invdepth_idx = net_eval(imgs, grids, test_mode=True)

                invdepth_idx = toNumpy(invdepth_idx[0, 0])

                # Compute errors
                errors[d, :] = data.evalError(invdepth_idx, gt, valid)
                mean_errors = errors[:d+1].mean(axis=0) 

                # LOG_INFO('Iter %d/%d >1: %.3f, >3: %.3f, >5: %.3f, MAE: %.3f, RMS: %.3f Time: %.2f' %
                #         (d, len(eval_list), mean_errors[0], mean_errors[1], mean_errors[2], mean_errors[3], mean_errors[4],time.time() - start_time_eval))
    
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        
        if args.stereo:
            measured = np.isfinite(errors).all(axis=1)
            mean_errors = errors[measured].mean(axis=0) if measured.any() else None
        else:
            mean_errors = errors.mean(axis=0)
        if mean_errors is None:
            LOG_INFO('Stereo validation skipped: no configured or valid validation samples')
        else:
            LOG_INFO('>1: %.3f, >3: %.3f, >5: %.3f, MAE: %.3f, RMS: %.3f' %
                (mean_errors[0], mean_errors[1], mean_errors[2], mean_errors[3], mean_errors[4]))

        # logging
        if args.wandb and not args.stereo:
            losses = dict()
            losses["val1"] = mean_errors[0]
            losses["val3"] = mean_errors[1]
            losses["val5"] = mean_errors[2]
            losses["val_mae"] = mean_errors[3]
            losses["val_rms"] = mean_errors[4]

            wandb.log({f"Eval/{name}": loss.item()
            for name, loss in losses.items()}, step=total_iters)

            # === VIS inputs: 반드시 detach + cpu ===
            raw_imgs_vis = [img[0].detach().cpu() for img in imgs]  # [768,800,3] x4
            # breakpoint()
            pred_vis = data.indexToInvdepth(invdepth_idx)           # [160,640] (마지막 prediction 권장)

            if torch.is_tensor(gt):
                gt_vis = data.indexToInvdepth(gt).detach().cpu() if gt.dim() == 3 else data.indexToInvdepth(gt).detach().cpu()
            else:
                gt_vis = data.indexToInvdepth(gt)  # gt가 list라면

            vis_results = data.makeVisImage(raw_imgs_vis, pred_vis, gt=gt_vis, return_all=True)
            wimages = {"Eval/Predictions": []}
            input_rgb, pred_rgb, gt_rgb, err_rgb = vis_results

            images = {
                f"Sample /Input": input_rgb,
                f"Sample /Prediction": pred_rgb,
                f"Sample /GT": gt_rgb,
                f"Sample /Error": err_rgb
            }
            wimages["Eval/Predictions"].extend(
                [wandb.Image(img, caption=caption) for caption, img in images.items()]
            )
            wandb.log(wimages, step=total_iters)
        
        # save
        savefilename = opts.model_dir + '/%s_e%d.pth' % (osp.basename(opts.name), epoch)
        snapshot = {
                'net_state_dict': net.state_dict(),
                'net_opts': opts.net_opts,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'epoch_loss': epoch_loss,
            }
        if args.stereo:
            snapshot.update(net_opts=net.module.opts, scheduler=scheduler.state_dict(),
                            scaler=scaler.state_dict(), global_step=total_iters,
                            optimizer_updates=optimizer_updates, stereo_geometry=data.metadata,
                            training_args=vars(args), bounded_smoke=bool(args.max_train_steps),
                            usable_batches=usable_batches,
                            validation_errors=mean_errors.tolist() if mean_errors is not None else None)
        torch.save(snapshot, savefilename)
        if args.stereo:
            summary = dict(epoch=epoch + 1, train_loss=float(epoch_loss), optimizer_updates=optimizer_updates,
                           validation_errors=mean_errors.tolist() if mean_errors is not None else None,
                           epoch_seconds=time.time()-epoch_start, batches=usable_batches,
                           peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                           peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, checkpoint=savefilename)
            with open(osp.join(opts.model_dir, 'epochs.jsonl'), 'a') as handle:
                handle.write(json.dumps(summary) + '\n')
            LOG_INFO('Epoch %d complete: %.1f s, checkpoint %s' % (epoch + 1, summary['epoch_seconds'], savefilename))
        if stop_training:
            break


def main():
    load_state = opts.snapshot_path is not None or opts.pretrain_path is not None
    train(opts.total_epochs, load_state)


if __name__ == "__main__":
    main()

