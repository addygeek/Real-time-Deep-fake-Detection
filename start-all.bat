@echo off
echo ========================================
echo SpectraShield - Starting All Services
echo ========================================

echo.
echo Checking if services are installed...

REM Check if backend node_modules exists
if not exist "backend\node_modules" (
    echo Installing backend dependencies...
    cd backend
    call npm install
    cd ..
)

REM Check if frontend node_modules exists
if not exist "frontend\node_modules" (
    echo Installing frontend dependencies...
    cd frontend
    call npm install
    cd ..
)

echo.
echo ========================================
echo Starting Services...
echo ========================================

echo.
echo [1/3] Starting Backend on port 4000...
start "SpectraShield Backend" cmd /k "cd backend && npm run dev"
timeout /t 3 /nobreak >nul

echo [2/3] Starting ML Engine on port 5000...
start "SpectraShield ML Engine" cmd /k "cd ml-engine && python api.py"
timeout /t 3 /nobreak >nul

echo [3/3] Frontend already running on port 3000...
echo.

echo ========================================
echo All Services Started!
echo ========================================
echo.
echo Access the application:
echo   Frontend: http://localhost:3000
echo   Backend:  http://localhost:4000
echo   ML API:   http://localhost:5000
echo.
echo Press any key to view service status...
pause >nul

echo.
echo Checking service health...
curl -s http://localhost:4000/health
echo.
curl -s http://localhost:5000/health
echo.

echo.
echo ========================================
echo Services are running!
echo Close this window to keep them running.
echo ========================================
pause
