@echo off
echo Starting Brain Server...
cd /d "%~dp0brain_server"
docker-compose up -d
echo.
echo Waiting for services to be ready...
timeout /t 5 /nobreak >nul
echo.
echo Checking status...
docker-compose ps
echo.
echo Testing health endpoint...
curl http://localhost:8000/health
echo.
echo.
echo Brain Server should now be running at http://localhost:8000
pause
