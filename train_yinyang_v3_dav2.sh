#!/usr/bin/env bash
# Yin-Yang + DA-V2 11-23 旗舰版启动脚本：
#   - Yin-Yang V2 (RGB + MeshGT_v1 + 去Refiner + Shallow Skip + Bilateral回投 + 置信度/分歧重叠选路 + Aux监督)
#   - 方案一 (Scheme I): --dav2_input_mode yinyang_rot3 (Yin/Yang 解畸变 + Yang 顺时针90度重力对齐送入冻结 DA-V2)
#   - 方案二 (Scheme II): --dav2_entropy_gate (基于 GEV 匹配熵 H_gev 的自适应先验门控融合)
#   - 优选几何损失 (Loss 3): --fisheye_normal_weight 0.5 --fisheye_edge_alpha 1.0
#
# 用法示例：
#   STAGE=FULL GPU_ID=0 BATCH_SIZE=2 GRAD_ACCUM=2 ./train_yinyang_v3_dav2.sh
#   STAGE=S12  GPU_ID=0 BATCH_SIZE=2 GRAD_ACCUM=2 ./train_yinyang_v3_dav2.sh
set -euo pipefail
cd "$(dirname "$0")"

STAGE="${STAGE:-FULL}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

YINYANG_CACHE="${YINYANG_CACHE:-/cache/adaptive_grid_v1_yin50_20261116/grid_v1_geometry}"
DB_ROOT="${DB_ROOT:-/root/lzx_ws/datasets/Omnidirectional_Stereo_Dataset}"
MESH_GT_ROOT="${MESH_GT_ROOT:-/root/lzx_ws/datasets/OmniThings_cam4_cam3_YinYang_MeshGT_v1}"
DAV2_CHART_SIZE="${DAV2_CHART_SIZE:-336}"

BASE_V2_ARGS=(
  --use_yinyang --yinyang_cache "$YINYANG_CACHE"
  --no_geometry_refiner
  --use_shallow_skip
  --yinyang_reproj_mode bilateral --yinyang_bilateral_tau 2.0
  --yinyang_overlap_mode confidence_disagree
  --yinyang_disagree_delta 3.0 --yinyang_disagree_gamma 0.8 --yinyang_disagree_beta_max 4.0
  --yinyang_aux_weight 0.25
)

STAGE_ARGS=()
case "$STAGE" in
  S1)
    DEFAULT_NAME="omnids_yinyang_v3_dav2_scheme1_rot3_20261123"
    STAGE_ARGS=(
      "${BASE_V2_ARGS[@]}"
      --dav2_input_mode yinyang_rot3 --dav2_chart_input_size "$DAV2_CHART_SIZE"
    )
    ;;
  S12)
    DEFAULT_NAME="omnids_yinyang_v3_dav2_scheme12_rot3_entgate_20261123"
    STAGE_ARGS=(
      "${BASE_V2_ARGS[@]}"
      --dav2_input_mode yinyang_rot3 --dav2_chart_input_size "$DAV2_CHART_SIZE"
      --dav2_entropy_gate
    )
    ;;
  FULL)
    DEFAULT_NAME="omnids_yinyang_v3_dav2_scheme12_loss3_20261123"
    STAGE_ARGS=(
      "${BASE_V2_ARGS[@]}"
      --dav2_input_mode yinyang_rot3 --dav2_chart_input_size "$DAV2_CHART_SIZE"
      --dav2_entropy_gate
      --fisheye_normal_weight 0.5 --fisheye_edge_alpha 1.0 --fisheye_normal_last_iters 4
    )
    ;;
  *)
    echo "Unknown STAGE='$STAGE' (expected S1, S12, or FULL)" >&2
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
