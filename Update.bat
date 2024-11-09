@@chcp 65001 > NUL
@echo off

git pull
timeout /t 3 /nobreak >nul
cls

set CURRENT_DIR=%~dp0

call "%CURRENT_DIR%venv\Scripts\activate.bat"

echo Setting up local pip cache...
if not exist "%CURRENT_DIR%TechnicalFiles\pip_cache" mkdir "%CURRENT_DIR%TechnicalFiles\pip_cache"
set PIP_CACHE_DIR=%CURRENT_DIR%TechnicalFiles\pip_cache

echo Updating dependencies...
if not exist "%CURRENT_DIR%TechnicalFiles\logs" mkdir "%CURRENT_DIR%TechnicalFiles\logs"
set ERROR_LOG="%CURRENT_DIR%TechnicalFiles\logs\update_errors.log"
type nul > %ERROR_LOG%

python -m pip install --upgrade pip
pip install wheel setuptools
pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements.txt" 2>> %ERROR_LOG%
pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-cuda.txt" 2>> %ERROR_LOG%
pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-llama-cpp.txt" 2>> %ERROR_LOG%
pip install --no-deps -r "%CURRENT_DIR%RequirementsFiles\requirements-stable-diffusion-cpp.txt" 2>> %ERROR_LOG%
timeout /t 3 /nobreak >nul
cls

echo Checking for update errors...
findstr /C:"error" %ERROR_LOG% >nul
if %ERRORLEVEL% equ 0 (
    echo Some packages failed to install. Please check %ERROR_LOG% for details.
) else (
    echo Installation completed successfully.
)
timeout /t 5 /nobreak >nul
cls

echo Application update process completed. Run start.bat to launch the application.

call "%CURRENT_DIR%venv\Scripts\deactivate.bat"

pause