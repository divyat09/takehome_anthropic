#!/usr/bin/env bash
# Sweep script for Topic A experiments.
# Run from takehome_20260128/ directory.
set -e

SCRIPT="run_topic_a.py"

echo "=========================================="
echo "Experiment 0: Baseline (default settings)"
echo "=========================================="
python "$SCRIPT" --exp_name baseline

echo ""
echo "=========================================="
echo "Experiment 1: Learning rate sweep"
echo "=========================================="
for LR in 1e-4 3e-4 1e-3 3e-3 1e-2; do
    echo "--- LR=$LR ---"
    python "$SCRIPT" --exp_name "lr_${LR}" --lr "$LR"
done

echo ""
echo "=========================================="
echo "Experiment 2: Batch size sweep"
echo "=========================================="
for BS in 64 256 1024 4096 16384; do
    echo "--- batch_size=$BS ---"
    python "$SCRIPT" --exp_name "bs_${BS}" --batch_size "$BS"
done


echo ""
echo "=========================================="
echo "Experiment 3: Teacher training epochs"
echo "=========================================="
for EP in 1 2 5 10 20; do
    echo "--- epochs_teacher=$EP ---"
    python "$SCRIPT" --exp_name "teacher_ep${EP}" --epochs_teacher "$EP"
done

echo ""
echo "All experiments complete. Results in results_a/, plots in plots_a/"

echo ""
echo "=========================================="
echo "Experiment 4: Weight decay sweep"
echo "=========================================="
for WD in 0.0 1e-4 1e-3 1e-2 0.1; do
    echo "--- weight_decay=$WD ---"
    python "$SCRIPT" --exp_name "wd_${WD}" --weight_decay "$WD"
done

echo ""
echo "=========================================="
echo "Experiment 5: Init distance (teacher-student noise)"
echo "=========================================="
for NOISE in 0.0 0.01 0.05 0.1 0.5 1.0; do
    echo "--- init_noise=$NOISE ---"
    python "$SCRIPT" --exp_name "initnoise_${NOISE}" --init_noise "$NOISE"
done


echo ""
echo "=========================================="
echo "Experiment 6: Distillation data distribution"
echo "=========================================="
for DIST in uniform gaussian shuffled_pixels; do
    echo "--- distill_data=$DIST ---"
    python "$SCRIPT" --exp_name "distill_${DIST}" --distill_data "$DIST"
done

