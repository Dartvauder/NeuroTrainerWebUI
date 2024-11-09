@echo off
chcp 65001 > NUL

set CURRENT_DIR=%~dp0
echo Creating virtual environment...
py -m venv "%CURRENT_DIR%venv"
call "%CURRENT_DIR%venv\Scripts\activate.bat"
cls

echo Setting up local pip cache...
if not exist "%CURRENT_DIR%TechnicalFiles\pip_cache" mkdir "%CURRENT_DIR%TechnicalFiles\pip_cache"
set PIP_CACHE_DIR=%CURRENT_DIR%TechnicalFiles\pip_cache

echo Upgrading pip, setuptools and wheel...
python -m pip install --upgrade pip
pip install wheel setuptools
timeout /t 3 /nobreak >nul
cls

echo Installing dependencies...
if not exist "%CURRENT_DIR%TechnicalFiles\logs" mkdir "%CURRENT_DIR%TechnicalFiles\logs"
set ERROR_LOG="%CURRENT_DIR%TechnicalFiles\logs\installation_errors.log"
type nul > %ERROR_LOG%

pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements.txt" 2>> %ERROR_LOG%
pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-cuda.txt" 2>> %ERROR_LOG%
pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-llama-cpp.txt" 2>> %ERROR_LOG%
pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-stable-diffusion-cpp.txt" 2>> %ERROR_LOG%
timeout /t 3 /nobreak >nul
cls

echo Checking for installation errors...
findstr /C:"error" %ERROR_LOG% >nul
if %ERRORLEVEL% equ 0 (
    echo Some packages failed to install. Please check %ERROR_LOG% for details.
) else (
    echo Installation completed successfully.
)
timeout /t 5 /nobreak >nul
cls

echo Application installation process completed. Run start.bat to launch the application.

call "%CURRENT_DIR%venv\Scripts\deactivate.bat"

pause
