#!/usr/bin/env bash
# Yin-Yang + DA-V2 RGB V2 渐进消融启动脚本（不影响原灰度脚本与历史权重）。
# 用法示例：
#   STAGE=A GPU_ID=0 ./train_yinyang_v2_rgb.sh   # 单网格 DA-V2 RGB 基线
#   STAGE=B GPU_ID=0 ./train_yinyang_v2_rgb.sh   # 干净 Yin-Yang RGB（去 Refiner）
#   STAGE=C GPU_ID=0 ./train_yinyang_v2_rgb.sh   # Stage B + Bilateral回投 + 置信度分歧选路 + 单路Aux Loss
#   STAGE=D GPU_ID=0 ./train_yinyang_v2_rgb.sh   # Stage C + CNN 浅深层门控融合
set -euo pipefail
cd "$(dirname "$0")"

STAGE="${STAGE:-D}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

YINYANG_CACHE="${YINYANG_CACHE:-/cache/adaptive_grid_v1_yin50_20261116/grid_v1_geometry}"
DB_ROOT="${DB_ROOT:-/root/lzx_ws/datasets/Omnidirectional_Stereo_Dataset}"
MESH_GT_ROOT="${MESH_GT_ROOT:-/root/lzx_ws/datasets/OmniThings_cam4_cam3_YinYang_MeshGT_v1}"

STAGE_ARGS=()
case "$STAGE" in
  A)
    DEFAULT_NAME="omnids_dav2_single_meshgt_rgb_scratch30_20261121"
    ;;
  B)
    DEFAULT_NAME="omnids_dav2_yinyang_meshgt_rgb_norefiner_scratch30_20261121"
    STAGE_ARGS=(
      --use_yinyang --yinyang_cache "$YINYANG_CACHE"
      --no_geometry_refiner
    )
    ;;
  C)
    DEFAULT_NAME="omnids_dav2_yinyang_meshgt_rgb_v2_edgeconf_scratch30_20261121"
    STAGE_ARGS=(
      --use_yinyang --yinyang_cache "$YINYANG_CACHE"
      --no_geometry_refiner
      --yinyang_reproj_mode bilateral --yinyang_bilateral_tau 2.0
      --yinyang_overlap_mode confidence_disagree
      --yinyang_disagree_delta 3.0 --yinyang_disagree_gamma 0.8 --yinyang_disagree_beta_max 4.0
      --yinyang_aux_weight 0.25
    )
    ;;
  D)
    DEFAULT_NAME="omnids_dav2_yinyang_meshgt_rgb_v2_edgeconf_skip_scratch30_20261121"
    STAGE_ARGS=(
      --use_yinyang --yinyang_cache "$YINYANG_CACHE"
      --no_geometry_refiner
      --use_shallow_skip
      --yinyang_reproj_mode bilateral --yinyang_bilateral_tau 2.0
      --yinyang_overlap_mode confidence_disagree
      --yinyang_disagree_delta 3.0 --yinyang_disagree_gamma 0.8 --yinyang_disagree_beta_max 4.0
      --yinyang_aux_weight 0.25
    )
    ;;
  *)
    echo "Unknown STAGE='$STAGE' (expected A, B, C, or D)" >&2
    exit 1
    ;;
esac

EXTRA_GT=()
if [[ -n "$MESH_GT_ROOT" ]]; then
  EXTRA_GT=(--mesh_gt_root "$MESH_GT_ROOT")
else
  EXTRA_GT=(--gt_width "${GT_WIDTH:-640}")
fi

EXTRA_RESTORE=()
if [[ -n "${RESTORE_CKPT:-}" ]]; then
  EXTRA_RESTORE=(--restore_ckpt "$RESTORE_CKPT")
elif [[ -n "${PRETRAIN_CKPT:-}" ]]; then
  EXTRA_RESTORE=(--pretrain_ckpt "$PRETRAIN_CKPT")
fi

exec "${PYTHON_BIN:-/opt/pdf-omni-venv/bin/python}" -u tools/retry_stereo_training.py \
  --stereo --ref_cam cam4 --target_cam cam3 --reference_grid left_fisheye \
  --use_rgb \
  --use_dav2_geometry --dav2_model_repo /cache/models/Depth-Anything-V2 \
  --dav2_checkpoint /cache/models/depth_anything_v2_vitb.pth \
  --db_root "$DB_ROOT" --dbname omnithings \
  "${EXTRA_GT[@]}" \
  --sweep_size 160 240 --num_invdepth 96 --num_depth_samples 96 \
  --total_epochs 30 --train_iters 12 --batch_size "${BATCH_SIZE:-4}" --num_workers 4 \
  --lr 0.0005 --wdecay 0.00001 --mixed_precision --amp_init_scale 128 \
  --gradient_checkpointing --val_count 512 --seed 0 \
  --name "${RUN_NAME:-$DEFAULT_NAME}" \
  "${STAGE_ARGS[@]}" \
  "${EXTRA_RESTORE[@]}" \
  "$@"
