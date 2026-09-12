@echo off
setlocal
rem ============================================================
rem  pixelart one-click launcher (Windows)
rem    1) ensure .venv exists and dependencies are importable
rem    2) ensure the depth model is downloaded (via hf-mirror)
rem    3) start the local server on http://127.0.0.1:8770
rem    4) open the default browser
rem  NOTE: keep this file PURE ASCII (same reason as scripts\setup.ps1 -
rem  cmd.exe reads .bat files with the system ANSI codepage).
rem ============================================================

cd /d "%~dp0"

set VPY=.venv\Scripts\python.exe
set PORT=8770

rem ---- 1) venv -------------------------------------------------
if exist "%VPY%" goto venv_ok
echo [setup] .venv not found - creating it ...
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 -m venv .venv
) else (
    python -m venv .venv
)
if not exist "%VPY%" (
    echo [error] Python 3.10+ not found. Install it from python.org first.
    pause
    exit /b 1
)
:venv_ok

rem ---- 2) dependencies ------------------------------------------
"%VPY%" -c "import numpy, PIL, cv2, onnxruntime" >nul 2>nul
if %errorlevel%==0 goto deps_ok
echo [setup] installing dependencies ...
"%VPY%" -m pip install -U pip
if exist requirements.txt (
    "%VPY%" -m pip install -r requirements.txt
) else (
    "%VPY%" -m pip install numpy pillow opencv-python-headless onnxruntime
)
"%VPY%" -c "import numpy, PIL, cv2, onnxruntime" >nul 2>nul
if errorlevel 1 (
    echo [error] dependency installation failed - see messages above.
    pause
    exit /b 1
)
:deps_ok
echo [ok] environment ready.

rem ---- 3) model --------------------------------------------------
if exist "models\depth-anything-v2-small\onnx\model.onnx" goto model_ok
echo [setup] depth model missing - downloading via hf-mirror ...
"%VPY%" tools\fetch_models.py
if not exist "models\depth-anything-v2-small\onnx\model.onnx" (
    echo [warn] model still missing - server will start, but rendering needs the model.
)
:model_ok
echo [ok] model ready.

rem ---- 4) server + browser ---------------------------------------
echo [run] starting server on http://127.0.0.1:%PORT% ...
start "PixelArt server" "%VPY%" tools\m3_server.py --port %PORT%

rem wait until the port answers (up to ~20s), then open the default browser
powershell -NoProfile -Command "for($i=0;$i -lt 40;$i++){try{$c=New-Object Net.Sockets.TcpClient('127.0.0.1',%PORT%);$c.Close();exit 0}catch{Start-Sleep -Milliseconds 500}}; exit 1"

start "" http://127.0.0.1:%PORT%
echo [done] browser opened. Close the server window to stop the app.
endlocal
