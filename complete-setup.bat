@echo off
echo ========================================
echo SpectraShield - Complete Setup
echo ========================================

echo.
echo This will:
echo  1. Install all dependencies
echo  2. Train the ML model
echo  3. Initialize the blockchain
echo  4. Start all services
echo.
pause

REM Backend Setup
echo.
echo [1/4] Setting up Backend...
cd backend
if not exist node_modules (
    echo Installing backend dependencies...
    call npm install
)
echo ✓ Backend ready
cd ..

REM Frontend Setup  
echo.
echo [2/4] Setting up Frontend...
cd frontend
if not exist node_modules (
    echo Installing frontend dependencies...
    call npm install
)
echo ✓ Frontend ready
cd ..

REM ML Engine Setup
echo.
echo [3/4] Setting up ML Engine...
cd ml-engine

echo Installing minimal Python dependencies...
pip install fastapi uvicorn pydantic numpy opencv-python tqdm

echo.
echo Checking for PyTorch...
python -c "import torch" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing PyTorch (CPU version - lightweight)...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

echo.
echo Creating pre-trained model...
python -c "from detector import create_pretrained_model; create_pretrained_model()"

echo.
echo Training model on synthetic data...
python train.py

echo ✓ ML Engine ready
cd ..

REM Initialize Blockchain
echo.
echo [4/4] Initializing Blockchain...
cd backend
node scripts\init-blockchain.js
cd ..

echo.
echo ========================================
echo ✓ Setup Complete!
echo ========================================
echo.
echo Starting all services...
echo.

REM Start services
start "Backend" cmd /k "cd backend && npm run dev"
timeout /t 2 /nobreak >nul

start "ML Engine" cmd /k "cd ml-engine && python api.py"
timeout /t 2 /nobreak >nul

start "Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================
echo All services started!
echo ========================================
echo.
echo Access the application at:
echo   http://localhost:3000
echo.
echo Close this window to keep services running.
pause
