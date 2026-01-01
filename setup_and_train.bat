@echo off
REM Automated setup and training script for Windows
REM This script will guide you through the setup process

echo ========================================
echo Dual-View Dog Image Matching - Setup
echo ========================================
echo.

REM Check if venv exists
if not exist "venv" (
    echo [1/6] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created!
) else (
    echo Virtual environment already exists, skipping...
)

echo.
echo [2/6] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo.
echo [3/6] Installing required packages...
echo This may take 5-15 minutes, please wait...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install packages
    pause
    exit /b 1
)
echo Packages installed successfully!

echo.
echo [4/6] Verifying installation...
python verify_installation.py
if errorlevel 1 (
    echo WARNING: Some checks failed, but continuing...
)

echo.
echo [5/6] Verifying dataset structure...
python verify_dataset.py data
if errorlevel 1 (
    echo WARNING: Dataset verification had issues, but continuing...
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Review STEP_BY_STEP_TRAINING.md for detailed instructions
echo 2. Run training with:
echo    python src/train.py --data_dir data --batch_size 16 --epochs 50
echo.
echo Or use the quick test command:
echo    python src/train.py --data_dir data --batch_size 8 --epochs 2
echo.
pause

