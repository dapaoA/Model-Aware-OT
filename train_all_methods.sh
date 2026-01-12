#!/bin/bash
# Training script for CIFAR-10 with three methods: OT-CFM, MA-TCFM, and CFM
# This script runs three training experiments sequentially

echo "========================================"
echo "Starting CIFAR-10 Training Experiments"
echo "========================================"
echo ""

# Common parameters
DATASET=cifar10
BATCH_SIZE=256
ITERATIONS=400000
LR=2e-4
SIGMA=0.0
SAVE_ITER=10000
LOG_ITER=1000
SEED=42

echo "========================================"
echo "Method 1/3: OT-CFM"
echo "========================================"
python train.py --method otcfm --dataset $DATASET --batch_size $BATCH_SIZE --iterations $ITERATIONS --lr $LR --sigma $SIGMA --save_iter $SAVE_ITER --log_iter $LOG_ITER --save_dir ./models/cifar10_otcfm --seed $SEED

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: OT-CFM training failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "Method 2/3: MA-TCFM"
echo "========================================"
python train.py --method ma_tcfm --dataset $DATASET --batch_size $BATCH_SIZE --iterations $ITERATIONS --lr $LR --sigma $SIGMA --save_iter $SAVE_ITER --log_iter $LOG_ITER --save_dir ./models/cifar10_ma_tcfm --seed $SEED

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: MA-TCFM training failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "Method 3/3: CFM"
echo "========================================"
python train.py --method cfm --dataset $DATASET --batch_size $BATCH_SIZE --iterations $ITERATIONS --lr $LR --sigma $SIGMA --save_iter $SAVE_ITER --log_iter $LOG_ITER --save_dir ./models/cifar10_cfm --seed $SEED

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: CFM training failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "All training experiments completed!"
echo "========================================"
echo ""
echo "Results saved in:"
echo "  - ./models/cifar10_otcfm"
echo "  - ./models/cifar10_ma_tcfm"
echo "  - ./models/cifar10_cfm"
echo ""
