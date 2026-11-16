#!/usr/bin/env bash
# 与原train.py共用优化器/LR/Sequence Loss；新增前端由冻结几何包驱动。
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
checkpoint_args=(--pretrain_ckpt "${PRETRAIN_CKPT:-/workspace/stereo_fisheye_omni_ds/checkpoints/omnids_dav2_context_resume30_20261113/omnids_dav2_context_resume30_20261113_e27.pth}")
if [[ -n "${RESTORE_CKPT:-}" ]]; then
  checkpoint_args=(--restore_ckpt "$RESTORE_CKPT")
fi
exec "${PYTHON_BIN:-/opt/pdf-omni-venv/bin/python}" -u train.py \
  --stereo --ref_cam cam4 --target_cam cam3 --reference_grid left_fisheye \
  --use_yinyang --yinyang_cache "${YINYANG_CACHE:-/cache/adaptive_grid_v1_yin50_20261116/grid_v1_geometry}" \
  --use_dav2_geometry --dav2_model_repo /cache/models/Depth-Anything-V2 \
  --dav2_checkpoint /cache/models/depth_anything_v2_vitb.pth \
  --db_root "${DB_ROOT:-/root/lzx_ws/datasets/Omnidirectional_Stereo_Dataset}" --dbname omnithings \
  --sweep_size 160 240 --num_invdepth 96 --num_depth_samples 96 \
  --total_epochs 30 --train_iters 12 --batch_size "${BATCH_SIZE:-2}" --num_workers 4 \
  --lr 0.0005 --wdecay 0.00001 --mixed_precision --amp_init_scale 128 \
  --gradient_checkpointing --val_count 512 --seed 0 \
  --name "${RUN_NAME:-omnids_dav2_yinyang_30ep_20261117}" \
  "${checkpoint_args[@]}" \
  "$@"
