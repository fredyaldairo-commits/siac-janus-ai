@echo off
chcp 65001 >nul
title JNUS AI - Panel de Administracion
color 0E
cd /d "%~dp0"

echo ============================================================
echo    JNUS AI - Panel de Administracion
echo ============================================================
echo.

REM --- Elegir un Python que tenga Flask instalado ---
set "PYEXE="
if exist ".venv\Scripts\python.exe" call :probar ".venv\Scripts\python.exe"
if not defined PYEXE call :probar "python"
if not defined PYEXE call :probar "py"
if not defined PYEXE goto sin_python

echo  [+] Python:    %PYEXE%
echo  [+] Usuario:   admin
echo  [+] Contrasena: jnus2026
echo.
echo  [!] NO cierres esta ventana: es el servidor.
echo  [+] El panel de admin se abrira solo en tu navegador...
echo.

REM Local de escritorio (NO produccion en la nube): arranque limpio, sin reloader,
REM admin sobre HTTP. La seguridad de produccion solo aplica en el host de la nube.
set "JNUS_LOCAL=1"
set "JNUS_OPEN_ADMIN=1"
%PYEXE% app.py

echo.
echo  [!] El servidor se ha detenido.
goto fin

:probar
"%~1" -c "import flask" 1>nul 2>nul
if not errorlevel 1 set "PYEXE=%~1"
goto :eof

:sin_python
echo  [X] No se encontro Python con Flask instalado.
echo.
echo      Instala las dependencias una sola vez con:
echo         python -m pip install -r requirements.txt
echo.

:fin
echo.
pause
