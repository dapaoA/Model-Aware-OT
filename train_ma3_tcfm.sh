#!/bin/bash
# Training script for MA3-TCFM (3x downsampling) on CIFAR-10

# Configuration
DATASET="cifar10"
BATCH_SIZE=256
ITERATIONS=400000
LR=2e-4
SIGMA=0.0
SAVE_ITER=10000
LOG_ITER=1000
SEED=42
SAVE_DIR="./models/cifar10_ma3_tcfm"

echo "=========================================="
echo "Training MA3-TCFM on CIFAR-10"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Batch size: $BATCH_SIZE"
echo "Iterations: $ITERATIONS"
echo "Learning rate: $LR"
echo "Sigma: $SIGMA"
echo "Save directory: $SAVE_DIR"
echo "=========================================="

# Activate virtual environment
source venv/bin/activate

# Run training
python train.py \
    --method ma3_tcfm \
    --dataset $DATASET \
    --batch_size $BATCH_SIZE \
    --iterations $ITERATIONS \
    --lr $LR \
    --sigma $SIGMA \
    --save_iter $SAVE_ITER \
    --log_iter $LOG_ITER \
    --save_dir $SAVE_DIR \
    --seed $SEED

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
    echo "Model saved to: $SAVE_DIR"
else
    echo ""
    echo "=========================================="
    echo "ERROR: Training failed!"
    echo "=========================================="
    exit 1
fi

