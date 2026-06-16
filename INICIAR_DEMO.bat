@echo off
title SenAlerta - Demo
echo ============================================
echo    SenAlerta - Iniciando demo
echo ============================================
echo.

rem si el backend ya esta corriendo, no arrancar otro
powershell -NoProfile -Command "try { Invoke-RestMethod -Uri http://localhost:8765/health -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if not errorlevel 1 (
  echo Backend ya esta corriendo.
  goto abrir
)

echo [1/3] Arrancando backend - tarda 1 a 2 minutos por TensorFlow...
start "SenAlerta Backend" cmd /k C:\v310\Scripts\python.exe "%~dp0python\server.py"

echo [2/3] Esperando a que el backend responda...
powershell -NoProfile -Command "$d=(Get-Date).AddSeconds(300); while((Get-Date) -lt $d){ try { Invoke-RestMethod -Uri http://localhost:8765/health -TimeoutSec 2 | Out-Null; exit 0 } catch { Start-Sleep -Seconds 3 } }; exit 1"
if errorlevel 1 (
  echo ERROR: el backend no respondio en 5 minutos. Revisa la ventana del backend.
  pause
  exit /b 1
)

:abrir
echo [3/3] Abriendo el popup de prueba...
start "" "%~dp0test_popup.html"
echo.
echo Listo. En el popup: boton "Encender camara" y a hacer senas.
echo Para el modo de grabacion del abecedario: boton "Grabacion en serie".
echo (No cierres la ventana del backend mientras dure la demo.)
pause
