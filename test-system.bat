@echo off
echo ========================================
echo SpectraShield - Quick Test
echo ========================================

echo.
echo Testing all components...
echo.

REM Test Backend
echo [1/3] Testing Backend...
curl -s http://localhost:4000/health
if %ERRORLEVEL% EQU 0 (
    echo ✓ Backend is working
) else (
    echo ✗ Backend is not responding
)

echo.
echo [2/3] Testing ML Engine...
curl -s http://localhost:5000/health
if %ERRORLEVEL% EQU 0 (
    echo ✓ ML Engine is working
) else (
    echo ✗ ML Engine is not responding
)

echo.
echo [3/3] Testing Frontend...
curl -s http://localhost:3000 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✓ Frontend is working
) else (
    echo ✗ Frontend is not responding
)

echo.
echo ========================================
echo Test Complete!
echo ========================================
echo.
echo If all tests passed, your system is fully operational!
echo.
echo Try uploading a video at: http://localhost:3000
echo.
pause
