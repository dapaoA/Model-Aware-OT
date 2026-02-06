@echo off
REM Windows批处理脚本：运行三个模型的配对误差实验
REM 模型：CFM, OTCFM, MA_TCFM
REM 请从项目根目录运行，或本脚本会自动切换到项目根目录
cd /d "%~dp0\.."

echo ========================================
echo Running Pairing Error Experiments
echo ========================================
echo.

REM 设置参数
set BATCH_SIZE=64
set NUM_T_SAMPLES=100
set DATA_DIR=./data
set OUTPUT_DIR=./exp/experiment_results

echo Configuration:
echo   Batch size: %BATCH_SIZE%
echo   Num t samples: %NUM_T_SAMPLES%
echo   Output directory: %OUTPUT_DIR%
echo.

REM 1. CFM模型
echo ========================================
echo [1/3] Running experiment for CFM model...
echo ========================================
python script/experiment_pairing_error.py ^
    --checkpoint models/cifar10_cfm/cfm_cifar10/checkpoint_iter_400000.pt ^
    --batch_size %BATCH_SIZE% ^
    --num_t_samples %NUM_T_SAMPLES% ^
    --data_dir %DATA_DIR% ^
    --output_dir %OUTPUT_DIR%

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CFM experiment failed!
    pause
    exit /b 1
)

echo.
echo CFM experiment completed successfully!
echo.

REM 2. OTCFM模型
echo ========================================
echo [2/3] Running experiment for OTCFM model...
echo ========================================
python script/experiment_pairing_error.py ^
    --checkpoint models/cifar10_otcfm/otcfm_cifar10/checkpoint_iter_400000.pt ^
    --batch_size %BATCH_SIZE% ^
    --num_t_samples %NUM_T_SAMPLES% ^
    --data_dir %DATA_DIR% ^
    --output_dir %OUTPUT_DIR%

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: OTCFM experiment failed!
    pause
    exit /b 1
)

echo.
echo OTCFM experiment completed successfully!
echo.

REM 3. MA_TCFM模型
echo ========================================
echo [3/3] Running experiment for MA_TCFM model...
echo ========================================
python script/experiment_pairing_error.py ^
    --checkpoint models/cifar10_ma_tcfm/ma_tcfm_cifar10/checkpoint_iter_400000.pt ^
    --batch_size %BATCH_SIZE% ^
    --num_t_samples %NUM_T_SAMPLES% ^
    --data_dir %DATA_DIR% ^
    --output_dir %OUTPUT_DIR%

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: MA_TCFM experiment failed!
    pause
    exit /b 1
)

echo.
echo MA_TCFM experiment completed successfully!
echo.

echo ========================================
echo All experiments completed successfully!
echo ========================================
echo.
echo Results saved to: %OUTPUT_DIR%
echo   - pairing_error_results_cfm_cifar10.npz
echo   - pairing_error_comparison_cfm_cifar10.png
echo   - pairing_error_results_otcfm_cifar10.npz
echo   - pairing_error_comparison_otcfm_cifar10.png
echo   - pairing_error_results_ma_tcfm_cifar10.npz
echo   - pairing_error_comparison_ma_tcfm_cifar10.png
echo.

pause
