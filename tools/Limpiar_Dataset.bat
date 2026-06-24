@echo off
chcp 65001 >nul
title JNUS AI - Limpiar Dataset
color 0B
echo.
echo  ============================================================
echo    JNUS AI - Limpiador de Datasets
echo    Prepara tu CSV/Excel antes de subirlo al panel /admin
echo  ============================================================
echo.
cd /d "%~dp0\.."

REM Usar el entorno virtual si existe
if exist ".venv\Scripts\python.exe" (
    set PY=.venv\Scripts\python.exe
) else (
    set PY=python
)

if "%~1"=="" (
    echo  Arrastra tu archivo de datos sobre este .bat,
    echo  o escribe la ruta a continuacion.
    echo.
    %PY% tools\limpiar_dataset.py
) else (
    %PY% tools\limpiar_dataset.py "%~1"
)

echo.
pause
