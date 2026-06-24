@echo off
chcp 65001 >nul
title Crear acceso directo de JNUS Admin
echo.
echo  Creando acceso directo "JNUS Admin" en tu Escritorio...
echo.
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$base = '%~dp0';" ^
  "$ws = New-Object -ComObject WScript.Shell;" ^
  "$desktop = [Environment]::GetFolderPath('Desktop');" ^
  "$dest = Join-Path $desktop 'JNUS Admin.lnk';" ^
  "$lnk = $ws.CreateShortcut($dest);" ^
  "$lnk.TargetPath = (Join-Path $base 'JNUS_Admin.bat');" ^
  "$lnk.WorkingDirectory = $base.TrimEnd('\');" ^
  "$lnk.Description = 'Abrir el Panel de Administracion de JNUS AI';" ^
  "$ico = Join-Path $base 'static\logo.ico'; if (Test-Path $ico) { $lnk.IconLocation = \"$ico,0\" } else { $lnk.IconLocation = \"$env:SystemRoot\System32\imageres.dll,109\" };" ^
  "$lnk.Save();" ^
  "if (Test-Path $dest) { Write-Host ''; Write-Host '  [OK] Acceso directo creado en:' -ForegroundColor Green; Write-Host \"        $dest\"; Write-Host '        Haz doble clic en el icono JNUS Admin cuando quieras.' } else { Write-Host '  [!] No se pudo crear. Usa directamente JNUS_Admin.bat' -ForegroundColor Yellow }"

echo.
pause
