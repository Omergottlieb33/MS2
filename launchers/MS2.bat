@echo off
rem Double-click to open the MS2 app. It runs on the lab server; this window keeps the
rem connection open. Settings are in ms2_settings.txt next to this file.
setlocal
for /f "usebackq eol=# tokens=1,* delims==" %%a in ("%~dp0ms2_settings.txt") do set "%%a=%%b"

echo Connecting to %HOST% as %SSH_USER% ...
echo The browser opens by itself once the app is up.
echo Close this window to stop the app. Analyses that are running keep going on the server.
echo.

rem Open the browser as soon as the app answers through the tunnel.
start "" /b powershell -NoProfile -WindowStyle Hidden -Command ^
  "for ($i = 0; $i -lt 90; $i++) { try { Invoke-WebRequest -UseBasicParsing http://localhost:%PORT%/ -TimeoutSec 1 | Out-Null; Start-Process http://localhost:%PORT%/; break } catch { Start-Sleep 1 } }"

ssh -t -o ExitOnForwardFailure=yes -L %PORT%:localhost:%PORT% %SSH_USER%@%HOST% "%MS2% app --port %PORT% --data-root %DATA_ROOT%"

echo.
echo The MS2 app has stopped. Analyses that were running keep going on the server.
pause
