@echo off
echo ========================================
echo SpectraShield Setup Script (Windows)
echo ========================================

echo Checking prerequisites...

where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Node.js is required but not installed.
    exit /b 1
)

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is required but not installed.
    exit /b 1
)

echo Prerequisites check complete

echo.
echo Setting up environment...
if not exist .env (
    copy .env.example .env
    echo Created .env file from template
) else (
    echo .env file already exists
)

echo.
echo Setting up backend...
cd backend
call npm install
echo Backend dependencies installed
cd ..

echo.
echo Setting up frontend...
cd frontend
call npm install
echo Frontend dependencies installed
cd ..

echo.
echo Setting up ML Engine...
cd ml-engine
pip install -r requirements.txt
echo ML Engine dependencies installed
cd ..

echo.
echo Creating directories...
if not exist backend\uploads mkdir backend\uploads
if not exist backend\data mkdir backend\data
if not exist ml-engine\models mkdir ml-engine\models
echo Directories created

echo.
echo Initializing blockchain...
cd backend
node scripts\init-blockchain.js
cd ..

echo.
echo ========================================
echo Setup complete!
echo.
echo To start the application:
echo   1. Start MongoDB
echo   2. Start Redis
echo   3. Start Backend: cd backend ^&^& npm run dev
echo   4. Start ML Engine: cd ml-engine ^&^& python api.py
echo   5. Start Frontend: cd frontend ^&^& npm run dev
echo.
echo Or use Docker: docker-compose up
echo.
pause
