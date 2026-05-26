@echo off
chcp 65001 >nul
echo ============================================================
echo   JNUS · Advanced Financial System
echo   Iniciando servidor Flask...
echo ============================================================
cd /d "%~dp0"

REM Crear venv si no existe
if not exist ".venv\Scripts\python.exe" (
    echo [+] Creando entorno virtual ".venv"...
    python -m venv .venv
)

REM Activar venv e instalar dependencias
call .venv\Scripts\activate.bat
echo [+] Instalando dependencias...
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet

REM Lanzar
echo [+] Servidor en http://127.0.0.1:5000
start "" http://127.0.0.1:5000
python app.py
pause
