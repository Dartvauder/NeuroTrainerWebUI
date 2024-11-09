## [Features](/#Features) | [Dependencies](/#Required-Dependencies) | [SystemRequirements](/#Minimum-System-Requirements) | [Install](/#How-to-install) | [Models](/#Where-can-I-get-models) | [Wiki](/#Wiki) | [Acknowledgment](/#Acknowledgment-to-developers) | [Licenses](/#Third-Party-Licenses)

![project-image](https://github.com/user-attachments/assets/2a47ff0d-9131-4c3b-897b-46f7cb9e4ae2)

* Work in progress! (ALPHA)
* English | [Русский](/Readmes/README_RU.md) | [漢語](/Readmes/README_ZH.md) 


## Description:

A simple and convenient interface for using of various neural network models. You can create datasets, finetune, evaluate and generate with LLM, StableDiffusion and StableAudio using various hyperparameters. You can also check the wiki, download the LLM, StableDiffusion and StableAudio models, change the application settings inside the interface and check system sensors

The goal of the project - to create the easiest possible application to finetune, evaluate and generate of neural network models

### LLM: <img width="1114" alt="1" src="https://github.com/user-attachments/assets/ed89c506-8b1d-49bb-8579-d1fdcfb94d9f">

### StableDiffusion: <img width="1111" alt="2" src="https://github.com/user-attachments/assets/19e21251-8fd1-4007-8ff6-7cfe1fff8f68">

### StableAudio: <img width="1116" alt="3" src="https://github.com/user-attachments/assets/d549715b-fb57-42c8-82c3-d34ad0977ec0">

### Interface: <img width="1111" alt="4" src="https://github.com/user-attachments/assets/5ac059af-5951-44b4-9fa4-85d1d8d83e42">

## Features:

* Easy installation via install.bat(Windows) or install.sh(Linux)
* You can use the application via your mobile device in localhost (Via IPv4) or anywhere online (Via Share)
* Flexible and optimized interface (By Gradio)
* Debug logging to logs from `Install` and `Update` files
* Available in three languages
* Support for Transformers: finetune, evaluate, quantize and generate (LLM)
* Support for Diffusers and Safetensors: finetune, evaluate, conversion, quantize and generate (StableDiffusion)
* Support for StableAudio: finetune and generate
* Full and LORA types of finetune, evaluate and generate (For LLM and StableDiffusion)
* Ability to create a dataset (For LLM, StableDiffusion and StableAudio models)
* Wiki
* ModelDownloader
* Application settings
* Ability to see system sensors

## Required Dependencies:

* [Python](https://www.python.org/downloads/) (3.10.11)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.4) and [cuDNN](https://developer.nvidia.com/cudnn-downloads) (9.1)
- C+ compiler
  - Windows: [VisualStudio](https://visualstudio.microsoft.com/ru/), [VisualStudioCode](https://code.visualstudio.com) and [Cmake](https://cmake.org)
  - Linux: [GCC](https://gcc.gnu.org/), [VisualStudioCode](https://code.visualstudio.com) and [Cmake](https://cmake.org)

## Minimum System Requirements:

* System: Windows or Linux
* GPU: 8GB+ or CPU: 16 core 3.6Ghz
* RAM: 24GB+
* Disk space: 10GB+
* Internet for installing

## How to install:

### Windows

1) First install all [RequiredDependencies](/#Required-Dependencies)
2) `Git clone https://github.com/Dartvauder/NeuroTrainerWebUI.git` to any location
3) Run the `Install.bat` and wait for installation
4) After installation, run `Start.bat`
5) Wait for the application to launch
6) Now you can start generating!

To get update, run `Update.bat`
To work with the virtual environment through the terminal, run `Venv.bat`

### Linux

1) First install all [RequiredDependencies](/#Required-Dependencies)
2) `Git clone https://github.com/Dartvauder/NeuroTrainerWebUI.git` to any location
3) In the terminal, run the `./Install.sh` and wait for installation of all dependencies
4) After installation, run `./Start.sh`
5) Wait for the application to launch
6) Now you can start generating!

To get update, run `./Update.sh`
To work with the virtual environment through the terminal, run `./Venv.sh`

## Wiki

* https://github.com/Dartvauder/NeuroTrainerWebUI/wiki

## Acknowledgment to developers

#### Many thanks to these projects because thanks to their applications/libraries, i was able to create my application:

First of all, I want to thank the developers of [PyCharm](https://www.jetbrains.com/pycharm/) and [GitHub](https://desktop.github.com). With the help of their applications, i was able to create and share my code

* `gradio` - https://github.com/gradio-app/gradio
* `transformers` - https://github.com/huggingface/transformers
* `diffusers` - https://github.com/huggingface/diffusers
* `llama.cpp-python` - https://github.com/abetlen/llama-cpp-python
* `stable-diffusion-cpp-python` - https://github.com/william-murray1204/stable-diffusion-cpp-python
* `stable-audio-tools` - https://github.com/Stability-AI/stable-audio-tools

## Third Party Licenses:

#### Many models have their own license for use. Before using it, I advise you to familiarize yourself with them:

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

#### These third-party repository codes are also used in my project:

* [Diffusers scripts for training SD](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)
* [StableAudioTools for training StableAudio](https://github.com/Stability-AI/stable-audio-tools)

## Donation

### *If you liked my project and want to donate, here is options to donate. Thank you very much in advance!*

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Dartvauder/NeuroTrainerWebUI&type=Date)](https://star-history.com/#Dartvauder/NeuroTrainerWebUI&Date)
