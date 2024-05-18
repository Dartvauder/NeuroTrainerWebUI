#!/bin/bash

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

echo "Creating virtual environment.../Создание виртуальной среды..."
python3 -m venv "$CURRENT_DIR/venv"
source "$CURRENT_DIR/venv/bin/activate"
clear

echo "Upgrading pip, setuptools and whell.../Обновление pip, setuptools и wheel..."
python3 -m pip install --upgrade pip setuptools
pip install wheel
clear

echo "Installing dependencies.../Установка зависимостей..."
pip install --no-deps -r "$CURRENT_DIR/requirements.txt"
clear

echo "Application has been installed successfully. Run start.sh/Приложение успешно установлено. Запустите start.sh"

deactivate

read -p "Press enter to continue/Нажмите enter для продолжения"
