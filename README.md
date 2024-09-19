## [Features](/#Features) | [Dependencies](/#Required-Dependencies) | [SystemRequirements](/#Minimum-System-Requirements) | [Install](/#How-to-install) | [Models](/#Where-can-I-get-models) | [Wiki](/#Wiki) | [Acknowledgment](/#Acknowledgment-to-developers) | [Licenses](/#Third-Party-Licenses)

![project-image](https://github.com/user-attachments/assets/2a47ff0d-9131-4c3b-897b-46f7cb9e4ae2)

* Work in progress! (ALPHA)
* English | [Русский](/Readmes/README_RU.md) | [漢語](/Readmes/README_ZH.md) 


## Description:

A simple and convenient interface for using of various neural network models. You can create datasets, finetune, evaluate and generate with LLM and StableDiffusion, using various hyperparameters. You can also check the wiki, download the LLM and StableDiffusion models, change the application settings inside the interface and check system sensors

The goal of the project - to create the easiest possible application to finetune, evaluate and generate of neural network models

### LLM: <img width="1118" alt="1" src="https://github.com/user-attachments/assets/683e7313-d6b2-45e8-8cbd-b919d9936123">

### StableDiffusion: <img width="1114" alt="2" src="https://github.com/user-attachments/assets/1ac84a3f-ad6f-44ad-bacf-fc491fbaadd6">

### Interface: <img width="1117" alt="3" src="https://github.com/user-attachments/assets/dfb8a606-6d11-4104-83fc-51a74a0b258a">

## Features:

* Easy installation via install.bat(Windows) or install.sh(Linux)
* Flexible and optimized interface (By Gradio)
* Authentication via admin:admin (You can enter your login details in the GradioAuth.txt file)
* Support for Transformers: finetune, evaluate, quantize and generate (LLM)
* Support for Diffusers and Safetensors: finetune, evaluate, conversion and generate (StableDiffusion)
* Full and LORA types of finetune, evaluate and generate (For LLM and StableDiffusion)
* Ability to create a dataset (For LLM and StableDiffusion)
* Wiki
* ModelDownloader (For LLM and StableDiffusion)
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

#### These third-party repository codes are also used in my project:

* [Diffusers scripts for training SD](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)

## Donation

### *If you liked my project and want to donate, here is options to donate. Thank you very much in advance!*

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Dartvauder/NeuroTrainerWebUI&type=Date)](https://star-history.com/#Dartvauder/NeuroTrainerWebUI&Date)
