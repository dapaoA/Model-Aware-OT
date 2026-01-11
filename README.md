# CIFAR10 Training with UNetExpert

## 文件说明

- `model/unet_expert.py`: UNetExpert模型，适配CIFAR10（3通道，32x32）
- `main.py`: 训练脚本，包含训练循环和FID计算

## 使用方法

1. 确保在conda huggingface环境下，安装所需依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 运行训练脚本：
   ```bash
   python main.py
   ```

## 配置说明

在`main.py`中可以调整以下参数：

- `batch_size`: 批次大小（默认64，RTX 3080 10GB可适当增加到128）
- `num_epochs`: 训练轮数（默认2用于测试，完整训练建议100+）
- `learning_rate`: 学习率（默认2e-4）
- `sigma`: CFM的sigma参数（默认0.0）

## 输出

- 模型保存路径: `models/cifar10_unet/model_epoch_{epoch}.pt`
- FID分数日志: `models/cifar10_unet/fid_scores.txt`

每个epoch结束后会自动：
1. 保存模型检查点
2. 计算FID分数（使用5000个生成样本）
3. 将FID分数记录到txt文件

## 注意事项

- FID计算需要一些时间，每个epoch可能需要几分钟
- 首次运行会下载CIFAR10数据集到`./data`目录
- FID计算使用cleanfid库，首次运行会下载预训练模型



# Training and Inference Scripts

This directory contains training and inference scripts for Conditional Flow Matching models.

## Files

- `train.py`: Training script with support for CFM, OT-CFM, and SB-CFM methods
- `infer.py`: Inference script for generating samples and computing FID scores
- `config/model_config.yaml`: Model architecture configuration for different datasets

## Usage

### Training

Train a model with default settings:

```bash
# Train CFM on moons dataset
python train.py --method cfm --dataset moons --iterations 20000 --save_iter 5000

# Train OT-CFM on CIFAR-10
python train.py --method otcfm --dataset cifar10 --batch_size 64 --iterations 100000 --save_iter 10000

# Train SB-CFM on 8gaussians dataset
python train.py --method sbcfm --dataset 8gaussians --sigma 0.5 --iterations 20000
```

#### Training Parameters

- `--method`: Training method - `cfm`, `otcfm`, or `sbcfm`
- `--dataset`: Dataset - `moons`, `8gaussians`, `cifar10`, or `mnist`
- `--model_config`: Path to model configuration YAML file (default: `config/model_config.yaml`)
- `--batch_size`: Batch size (default: 256)
- `--iterations`: Total training iterations (default: 20000)
- `--lr`: Learning rate (default: 1e-3)
- `--sigma`: Sigma parameter for flow matching (default: 0.1)
- `--save_dir`: Directory to save models (default: `./models`)
- `--save_iter`: Save checkpoint every N iterations (default: 5000)
- `--log_iter`: Log loss every N iterations (default: 1000)
- `--vis_steps`: Number of denoising steps for visualization (default: 50)
- `--vis_step_interval`: Interval between visualized steps (default: 5)
- `--seed`: Random seed (default: 42)

### Inference

Generate samples from a trained model:

```bash
# Generate samples
python infer.py --checkpoint ./models/checkpoint_iter_20000.pt --num_samples 1000 --num_steps 50

# Generate samples and compute FID (for image datasets)
python infer.py --checkpoint ./models/checkpoint_iter_20000.pt --num_samples 5000 --num_steps 50 --compute_fid
```

#### Inference Parameters

- `--checkpoint`: Path to checkpoint file (required)
- `--dataset`: Dataset name (optional, will use from checkpoint if available)
- `--num_samples`: Number of samples to generate (default: 1000)
- `--num_steps`: Number of ODE steps for generation (default: 50)
- `--output_dir`: Directory to save generated samples (default: `./generated_samples`)
- `--compute_fid`: Compute FID score (requires clean-fid and image dataset)
- `--seed`: Random seed (default: 42)

## Model Configuration

The `config/model_config.yaml` file defines model architectures for different datasets:

- **2D datasets** (moons, 8gaussians): MLP with configurable width
- **Image datasets** (cifar10, mnist): UNet with configurable channels and depth

You can modify this file to adjust model architectures for your needs.

## Examples

### Example 1: Train CFM on Moons Dataset

```bash
python train.py \
    --method cfm \
    --dataset moons \
    --batch_size 256 \
    --iterations 20000 \
    --save_iter 5000 \
    --seed 42
```

### Example 2: Train OT-CFM on CIFAR-10

```bash
python train.py \
    --method otcfm \
    --dataset cifar10 \
    --batch_size 64 \
    --iterations 100000 \
    --save_iter 10000 \
    --lr 2e-4 \
    --sigma 0.0 \
    --seed 42
```

### Example 3: Generate Samples and Compute FID

```bash
python infer.py \
    --checkpoint ./models/checkpoint_iter_100000.pt \
    --num_samples 5000 \
    --num_steps 50 \
    --compute_fid \
    --output_dir ./generated_cifar10
```

## Notes

- For image datasets (CIFAR-10, MNIST), the script uses UNet architecture
- For 2D datasets (moons, 8gaussians), the script uses MLP architecture
- FID computation requires `clean-fid` package: `pip install clean-fid`
- FID is only supported for image datasets (CIFAR-10, MNIST)
- The visualization shows denoising process: rows = different noise samples, columns = denoising steps

## Output Structure

After training, the save directory will contain:
- `checkpoint_iter_*.pt`: Model checkpoints
- `denoising_iter_*.png`: Visualization of denoising process

After inference, the output directory will contain:
- `generated_samples_grid.png`: Grid of generated samples
- `sample_*.png`: Individual sample images (for image datasets)
- `generated_samples.png`: Scatter plot (for 2D datasets)
- `fid_score.txt`: FID score (if computed)
