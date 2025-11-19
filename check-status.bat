@echo off
echo ========================================
echo SpectraShield - System Status Check
echo ========================================
echo.

echo Checking Frontend (Port 3000)...
curl -s http://localhost:3000 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Frontend is running
) else (
    echo [X] Frontend is NOT running
)

echo.
echo Checking Backend (Port 4000)...
curl -s http://localhost:4000/health
if %ERRORLEVEL% EQU 0 (
    echo [OK] Backend is running
) else (
    echo [X] Backend is NOT running
)

echo.
echo Checking ML Engine (Port 5000)...
curl -s http://localhost:5000/health
if %ERRORLEVEL% EQU 0 (
    echo [OK] ML Engine is running
) else (
    echo [X] ML Engine is NOT running
)

echo.
echo ========================================
echo System Status Summary
echo ========================================
echo.
echo If all services show [OK], your system is ready!
echo.
echo Access Points:
echo   Frontend:  http://localhost:3000
echo   Backend:   http://localhost:4000
echo   ML Engine: http://localhost:5000
echo.
echo API Documentation:
echo   Backend:   http://localhost:4000/health
echo   ML Engine: http://localhost:5000
echo.
pause
