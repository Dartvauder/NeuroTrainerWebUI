## [Функции](/#Функции) | [Зависимости](/#Необходимые-зависимости) | [Системные требования](/#Минимальные-системные-требования) | [Установка](/#Как-установить) | [Модели](/#Где-я-могу-получить-модели) | [Вики](/#Вики) | [Благодарность](/#Благодарность-разработчикам) | [Лицензии](/#Сторонние-лицензии)

![изображение-проекта](https://github.com/user-attachments/assets/2a47ff0d-9131-4c3b-897b-46f7cb9e4ae2)

* В разработке! (АЛЬФА)
* [English](/README.md) | Русский | [漢語](/TechnicalFiles/Readmes/README_ZH.md) 

## Описание:

Простой и удобный интерфейс для использования различных моделей нейронных сетей. Вы можете создавать наборы данных, проводить дообучение, оценивать и генерировать с помощью LLM, StableDiffusion и StableAudio, используя различные гиперпараметры. Вы также можете просмотреть вики, скачать модели LLM, StableDiffusion и StableAudio, изменить настройки приложения внутри интерфейса и проверить системные датчики.

Цель проекта - создать максимально простое приложение для дообучения, оценки и генерации моделей нейронных сетей.

### LLM: <img width="1115" alt="1ru" src="https://github.com/user-attachments/assets/d60252a6-18b1-40a3-a8bd-dd14420b4f37">

### StableDiffusion: <img width="1115" alt="2ru" src="https://github.com/user-attachments/assets/82e9f866-cded-482f-9d5b-94082c791fa5">

### StableAudio: <img width="1115" alt="3ru" src="https://github.com/user-attachments/assets/197f8c46-4056-44bf-bf65-f294b614aa76">

### Интерфейс: <img width="1114" alt="4ru" src="https://github.com/user-attachments/assets/39cdc4d5-efd5-440c-9c00-6c163452b0d0">

## Функции:

* Простая установка через install.bat (Windows) или install.sh (Linux)
* Гибкий и оптимизированный интерфейс (с помощью Gradio)
* Поддержка Transformers: дообучение, оценка, квантизация и генерация (LLM)
* Поддержка Diffusers и Safetensors: дообучение, оценка, конвертация, квантизация и генерация (StableDiffusion)
* Поддержка StableAudio: дообучение и генерация
* Полный и LORA типы дообучения, оценки и генерации (для LLM и StableDiffusion)
* Возможность создания набора данных (для моделей LLM, StableDiffusion и StableAudio)
* Вики
* Загрузчик моделей (для LLM и StableDiffusion)
* Настройки приложения
* Возможность просмотра системных датчиков

## Необходимые зависимости:

* [Python](https://www.python.org/downloads/) (3.10.11)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.4) и [cuDNN](https://developer.nvidia.com/cudnn-downloads) (9.1)
- Компилятор C++
  - Windows: [VisualStudio](https://visualstudio.microsoft.com/ru/), [VisualStudioCode](https://code.visualstudio.com) и [Cmake](https://cmake.org)
  - Linux: [GCC](https://gcc.gnu.org/), [VisualStudioCode](https://code.visualstudio.com) и [Cmake](https://cmake.org)

## Минимальные системные требования:

* Система: Windows или Linux
* GPU: 8 ГБ+ или CPU: 16 ядер 3.6 ГГц
* ОЗУ: 24 ГБ+
* Дисковое пространство: 10 ГБ+
* Интернет для установки

## Как установить:

### Windows

1) Сначала установите все [Необходимые зависимости](/#Необходимые-зависимости)
2) `Git clone https://github.com/Dartvauder/NeuroTrainerWebUI.git` в любое место
3) Запустите `Install.bat` и дождитесь завершения установки
4) После установки запустите `Start.bat`
5) Дождитесь запуска приложения
6) Теперь вы можете начать генерацию!

Для получения обновлений запустите `Update.bat`
Для работы с виртуальным окружением через терминал запустите `Venv.bat`

### Linux

1) Сначала установите все [Необходимые зависимости](/#Необходимые-зависимости)
2) `Git clone https://github.com/Dartvauder/NeuroTrainerWebUI.git` в любое место
3) В терминале запустите `./Install.sh` и дождитесь установки всех зависимостей
4) После установки запустите `./Start.sh`
5) Дождитесь запуска приложения
6) Теперь вы можете начать генерацию!

Для получения обновлений запустите `./Update.sh`
Для работы с виртуальным окружением через терминал запустите `./Venv.sh`

## Вики

* https://github.com/Dartvauder/NeuroTrainerWebUI/wiki

## Благодарность разработчикам

#### Большое спасибо этим проектам, потому что благодаря их приложениям/библиотекам я смог создать свое приложение:

Прежде всего, я хочу поблагодарить разработчиков [PyCharm](https://www.jetbrains.com/pycharm/) и [GitHub](https://desktop.github.com). С помощью их приложений я смог создать и поделиться своим кодом.

* `gradio` - https://github.com/gradio-app/gradio
* `transformers` - https://github.com/huggingface/transformers
* `diffusers` - https://github.com/huggingface/diffusers
* `llama.cpp-python` - https://github.com/abetlen/llama-cpp-python
* `stable-diffusion-cpp-python` - https://github.com/william-murray1204/stable-diffusion-cpp-python

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
* [StableAudioOpen](https://huggingface.co/stabilityai/stable-audio-open-1.0)
* [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-large)

#### Эти сторонние репозитории кода также используются в моем проекте:

* [Скрипты Diffusers для обучения SD](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)
* [StableAudioTools для обучения StableAudio](https://github.com/Stability-AI/stable-audio-tools)

## Пожертвование

### *Если вам понравился мой проект и вы хотите сделать пожертвование, вот варианты для пожертвования. Заранее большое спасибо!*

* [!["Купить мне кофе"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)

## История звезд

[![График истории звезд](https://api.star-history.com/svg?repos=Dartvauder/NeuroTrainerWebUI&type=Date)](https://star-history.com/#Dartvauder/NeuroTrainerWebUI&Date)
