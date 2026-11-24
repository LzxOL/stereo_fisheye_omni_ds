#!/usr/bin/env bash
# Yin-Yang + DA-V2 11-25 架构定型启动脚本 (统一 Smooth L1 先定架构，再定 Loss):
#   - STAGE=S1       : 任务一 — 纯方案一 (yinyang_rot3) + Smooth L1 基准对照
#   - STAGE=M1       : 消融   — 方案一 + M1 (DA-V2 GEV引导 + L_init后验校准 + d_0初值)
#   - STAGE=M12      : 消融   — 方案一 + M1 + M2 (语义/几何双通路解耦 + 单调有界门控)
#   - STAGE=V2_FULL  : 任务二 — 方案一 + 新方案二完整版 (M1 + M2 + M3) + Smooth L1
#   - STAGE=V2_LOSS3 : 可选   — 新方案二完整版 + Loss 3 (3D法向 + 边缘加权)
#
# 用法示例：
#   STAGE=S1      GPU_ID=1 BATCH_SIZE=1 GRAD_ACCUM=4 STOP_AFTER_EPOCH=10 ./train_yinyang_v4_dav2.sh
#   STAGE=V2_FULL GPU_ID=0 BATCH_SIZE=2 GRAD_ACCUM=2 STOP_AFTER_EPOCH=10 ./train_yinyang_v4_dav2.sh
set -euo pipefail
cd "$(dirname "$0")"

STAGE="${STAGE:-V2_FULL}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

YINYANG_CACHE="${YINYANG_CACHE:-/cache/adaptive_grid_v1_yin50_20261116/grid_v1_geometry}"
DB_ROOT="${DB_ROOT:-/root/lzx_ws/datasets/Omnidirectional_Stereo_Dataset}"
MESH_GT_ROOT="${MESH_GT_ROOT:-/root/lzx_ws/datasets/OmniThings_cam4_cam3_YinYang_MeshGT_v1}"
DAV2_CHART_SIZE="${DAV2_CHART_SIZE:-336}"
GEV_INIT_WEIGHT="${GEV_INIT_WEIGHT:-0.3}"

BASE_V2_ARGS=(
  --use_yinyang --yinyang_cache "$YINYANG_CACHE"
  --no_geometry_refiner
  --use_shallow_skip
  --yinyang_reproj_mode bilateral --yinyang_bilateral_tau 2.0
  --yinyang_overlap_mode confidence_disagree
  --yinyang_disagree_delta 3.0 --yinyang_disagree_gamma 0.8 --yinyang_disagree_beta_max 4.0
  --yinyang_aux_weight 0.25
  --dav2_input_mode yinyang_rot3 --dav2_chart_input_size "$DAV2_CHART_SIZE"
)

M1_ARGS=(
  --dav2_gev_guide
  --gev_init_disp
  --gev_init_loss_weight "$GEV_INIT_WEIGHT"
)
M2_ARGS=(
  --dav2_decoupled_fusion
)
M3_ARGS=(
  --gru_entropy_arbitration
  --yinyang_entropy_overlap
)

STAGE_ARGS=()
case "$STAGE" in
  S1)
    DEFAULT_NAME="omnids_yinyang_v4_dav2_s1_smoothl1_20261125"
    STAGE_ARGS=("${BASE_V2_ARGS[@]}")
    ;;
  M1)
    DEFAULT_NAME="omnids_yinyang_v4_dav2_s1_m1_smoothl1_20261125"
    STAGE_ARGS=("${BASE_V2_ARGS[@]}" "${M1_ARGS[@]}")
    ;;
  M12)
    DEFAULT_NAME="omnids_yinyang_v4_dav2_s1_m12_smoothl1_20261125"
    STAGE_ARGS=("${BASE_V2_ARGS[@]}" "${M1_ARGS[@]}" "${M2_ARGS[@]}")
    ;;
  V2_FULL)
    DEFAULT_NAME="omnids_yinyang_v4_dav2_s1_m123_smoothl1_20261125"
    STAGE_ARGS=("${BASE_V2_ARGS[@]}" "${M1_ARGS[@]}" "${M2_ARGS[@]}" "${M3_ARGS[@]}")
    ;;
  V2_LOSS3)
    DEFAULT_NAME="omnids_yinyang_v4_dav2_s1_m123_loss3_20261125"
    STAGE_ARGS=(
      "${BASE_V2_ARGS[@]}" "${M1_ARGS[@]}" "${M2_ARGS[@]}" "${M3_ARGS[@]}"
      --fisheye_normal_weight 0.5 --fisheye_edge_alpha 1.0 --fisheye_normal_last_iters 4
    )
    ;;
  *)
    echo "Unknown STAGE='$STAGE' (expected S1, M1, M12, V2_FULL, or V2_LOSS3)" >&2
    exit 1
    ;;
esac

EXTRA_ARGS=()
if [[ -n "${STOP_AFTER_EPOCH:-}" ]]; then
  EXTRA_ARGS+=(--stop_after_epoch "$STOP_AFTER_EPOCH")
fi
if [[ -n "${RESTORE_CKPT:-}" ]]; then
  EXTRA_ARGS+=(--restore_ckpt "$RESTORE_CKPT")
elif [[ -n "${PRETRAIN_CKPT:-}" ]]; then
  EXTRA_ARGS+=(--pretrain_ckpt "$PRETRAIN_CKPT")
fi

exec "${PYTHON_BIN:-/opt/pdf-omni-venv/bin/python}" -u train.py \
  --stereo --ref_cam cam4 --target_cam cam3 --reference_grid left_fisheye \
  --use_rgb \
  --use_dav2_geometry --dav2_model_repo /cache/models/Depth-Anything-V2 \
  --dav2_checkpoint /cache/models/depth_anything_v2_vitb.pth \
  --db_root "$DB_ROOT" --dbname omnithings \
  --mesh_gt_root "$MESH_GT_ROOT" \
  --sweep_size 160 240 --num_invdepth 96 --num_depth_samples 96 \
  --total_epochs "${TOTAL_EPOCHS:-30}" --train_iters 12 \
  --batch_size "${BATCH_SIZE:-2}" --gradient_accumulation "${GRAD_ACCUM:-2}" --num_workers 4 \
  --lr 0.0005 --wdecay 0.00001 --mixed_precision --amp_init_scale 128 \
  --gradient_checkpointing --val_count 512 --seed 0 \
  --name "${RUN_NAME:-$DEFAULT_NAME}" \
  "${STAGE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
