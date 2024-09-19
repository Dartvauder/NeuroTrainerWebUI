# Веб-интерфейс для обучения нейронных сетей

## [Функции](/#Функции) | [Зависимости](/#Необходимые-зависимости) | [Системные требования](/#Минимальные-системные-требования) | [Установка](/#Как-установить) | [Модели](/#Где-можно-получить-модели) | [Wiki](/#Wiki) | [Благодарность](/#Благодарность-разработчикам) | [Лицензии](/#Сторонние-лицензии)

![project-image](https://github.com/user-attachments/assets/2a47ff0d-9131-4c3b-897b-46f7cb9e4ae2)

* В процессе разработки! (АЛЬФА-версия)
* [English](/README.md) | Русский | [漢語](/Readmes/README_ZH.md)

## Описание:

Простой и удобный интерфейс для использования различных моделей нейронных сетей. Вы можете создавать датасеты, дообучать, оценивать и генерировать с помощью LLM и StableDiffusion, используя различные гиперпараметры. Вы также можете проверить wiki, загрузить модели LLM и StableDiffusion, изменить настройки приложения внутри интерфейса и проверить системные датчики.

Цель проекта - создать максимально простое приложение для дообучения, оценки и генерации моделей нейронных сетей.

### LLM: 

### StableDiffusion: 

### Интерфейс: 

## Функции:

* Простая установка через install.bat (Windows) или install.sh (Linux)
* Гибкий и оптимизированный интерфейс (на основе Gradio)
* Аутентификация через admin:admin (Вы можете ввести свои данные для входа в файл GradioAuth.txt)
* Поддержка Transformers: дообучение, оценка, квантизация и генерация (LLM)
* Поддержка Diffusers и Safetensors: дообучение, оценка, конвертация и генерация (StableDiffusion)
* Полный и LORA типы дообучения, оценки и генерации (для LLM и StableDiffusion)
* Возможность создания датасета (для LLM и StableDiffusion)
* Wiki
* ModelDownloader (для LLM и StableDiffusion)
* Настройки приложения
* Возможность просмотра системных датчиков

## Необходимые зависимости:

* [Python](https://www.python.org/downloads/) (3.10.11)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.4) и [cuDNN](https://developer.nvidia.com/cudnn-downloads) (9.1)
- C++ компилятор
  - Windows: [VisualStudio](https://visualstudio.microsoft.com/ru/), [VisualStudioCode](https://code.visualstudio.com) и [Cmake](https://cmake.org)
  - Linux: [GCC](https://gcc.gnu.org/), [VisualStudioCode](https://code.visualstudio.com) и [Cmake](https://cmake.org)

## Минимальные системные требования:

* Система: Windows или Linux
* GPU: 8GB+ или CPU: 16 ядер 3.6Ghz
* RAM: 24GB+
* Дисковое пространство: 10GB+
* Интернет для установки

## Как установить:

### Windows

1) Сначала установите все [Необходимые зависимости](/#Необходимые-зависимости)
2) Выполните `Git clone https://github.com/Dartvauder/NeuroTrainerWebUI.git` в любое место
3) Запустите `Install.bat` и дождитесь завершения установки
4) После установки запустите `Start.bat`
5) Дождитесь запуска приложения
6) Теперь вы можете начать генерацию!

Для получения обновлений запустите `Update.bat`
Для работы с виртуальным окружением через терминал запустите `Venv.bat`

### Linux

1) Сначала установите все [Необходимые зависимости](/#Необходимые-зависимости)
2) Выполните `Git clone https://github.com/Dartvauder/NeuroTrainerWebUI.git` в любое место
3) В терминале запустите `./Install.sh` и дождитесь установки всех зависимостей
4) После установки запустите `./Start.sh`
5) Дождитесь запуска приложения
6) Теперь вы можете начать генерацию!

Для получения обновлений запустите `./Update.sh`
Для работы с виртуальным окружением через терминал запустите `./Venv.sh`

## Wiki

* https://github.com/Dartvauder/NeuroTrainerWebUI/wiki

## Благодарность разработчикам

#### Большое спасибо этим проектам, потому что благодаря их приложениям/библиотекам я смог создать свое приложение:

Прежде всего, я хочу поблагодарить разработчиков [PyCharm](https://www.jetbrains.com/pycharm/) и [GitHub](https://desktop.github.com). С помощью их приложений я смог создать и поделиться своим кодом.

* `gradio` - https://github.com/gradio-app/gradio
* `transformers` - https://github.com/huggingface/transformers
* `diffusers` - https://github.com/huggingface/diffusers

## Сторонние лицензии:

#### Многие модели имеют свои собственные лицензии на использование. Перед использованием я советую вам ознакомиться с ними:

* [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE)
* [Diffusers](https://github.com/huggingface/diffusers/blob/main/LICENSE)
* [StableDiffusion1.5](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
* [StableDiffusion2](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
* [StableDiffusionXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
* [CLIP](https://huggingface.co/openai/clip-vit-base-patch16)
* [BERT](https://huggingface.co/google-bert/bert-base-uncased)
* [LLAMA.CPP](https://github.com/ggerganov/llama.cpp/blob/master/LICENSE)

#### Эти коды сторонних репозиториев также используются в моем проекте:

* [Скрипты Diffusers для обучения SD](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)

## Пожертвование

### *Если вам понравился мой проект и вы хотите сделать пожертвование, вот варианты для этого. Заранее большое спасибо!*

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Dartvauder/NeuroTrainerWebUI&type=Date)](https://star-history.com/#Dartvauder/NeuroTrainerWebUI&Date)
