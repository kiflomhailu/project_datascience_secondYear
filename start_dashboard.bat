@echo off
REM Seismic Dashboard Launcher Script (Windows)

echo ==================================
echo 🚦 Seismic Risk Dashboard Launcher
echo ==================================
echo.

REM Check if dashboard folder exists
if not exist "dashboard" (
    echo ❌ Error: dashboard folder not found
    echo    Please run this script from latest_cop directory
    pause
    exit /b 1
)

REM Start API server
echo 📡 Starting Flask API server...
cd dashboard\api
start "Flask API" cmd /c "python app.py"
cd ..\..

REM Wait for API to start
echo ⏳ Waiting for API to be ready...
timeout /t 5 /nobreak >nul

REM Start dashboard server
echo 🌐 Starting dashboard HTTP server...
cd dashboard
start "Dashboard" cmd /c "python -m http.server 8080"
cd ..

echo.
echo ==================================
echo ✅ Dashboard is running!
echo ==================================
echo.
echo 📊 Dashboard: http://localhost:8080
echo 📡 API:       http://localhost:5000
echo.
echo Press Ctrl+C in the terminal windows to stop
echo ==================================
echo.

REM Open browser
timeout /t 2 /nobreak >nul
start http://localhost:8080

pause
