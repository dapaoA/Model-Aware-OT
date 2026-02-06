@echo off
REM Windows批处理脚本：训练使用logit_normal和transport_logit_normal时间采样的OT-CFM模型

echo ========================================
echo Training OT-CFM with Logit-Normal Time Sampling
echo ========================================
echo.

REM 设置参数
set DATASET=cifar10
set BATCH_SIZE=128
set ITERATIONS=20000
set LR=1e-3
set SIGMA=0.1
set SAVE_ITER=5000
set LOG_ITER=1000

echo Configuration:
echo   Dataset: %DATASET%
echo   Batch size: %BATCH_SIZE%
echo   Iterations: %ITERATIONS%
echo   Learning rate: %LR%
echo   Sigma: %SIGMA%
echo.

REM 1. OT-CFM with logit_normal time sampling
echo ========================================
echo [1/2] Training OT-CFM with logit_normal time sampling...
echo ========================================
python train_logit_normal_otcfm.py ^
    --method otcfm ^
    --time_sampler logit_normal ^
    --dataset %DATASET% ^
    --batch_size %BATCH_SIZE% ^
    --iterations %ITERATIONS% ^
    --lr %LR% ^
    --sigma %SIGMA% ^
    --save_iter %SAVE_ITER% ^
    --log_iter %LOG_ITER% ^
    --save_dir ./models

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: OT-CFM with logit_normal training failed!
    pause
    exit /b 1
)

echo.
echo OT-CFM with logit_normal training completed successfully!
echo.

REM 2. OT-CFM with transport_logit_normal time sampling
echo ========================================
echo [2/2] Training OT-CFM with transport_logit_normal time sampling...
echo ========================================
python train_logit_normal_otcfm.py ^
    --method otcfm ^
    --time_sampler transport_logit_normal ^
    --dataset %DATASET% ^
    --batch_size %BATCH_SIZE% ^
    --iterations %ITERATIONS% ^
    --lr %LR% ^
    --sigma %SIGMA% ^
    --save_iter %SAVE_ITER% ^
    --log_iter %LOG_ITER% ^
    --save_dir ./models

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: OT-CFM with transport_logit_normal training failed!
    pause
    exit /b 1
)

echo.
echo OT-CFM with transport_logit_normal training completed successfully!
echo.

echo ========================================
echo All training completed successfully!
echo ========================================
echo.
echo Models saved to: ./models
echo   - otcfm_cifar10_logit_normal/
echo   - otcfm_cifar10_transport_logit_normal/
echo.

pause
