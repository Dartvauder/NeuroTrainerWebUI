## [Features](/#Features) | [Dependencies](/#Required-Dependencies) | [SystemRequirements](/#Minimum-System-Requirements) | [Install](/#How-to-install) | [Usage](/#How-to-use) | [Models](/#Where-can-I-get-models) | [Wiki](/#Wiki) | [Acknowledgment](/#Acknowledgment-to-developers) | [Licenses](/#Third-Party-Licenses)

* Work in progress! (ALPHA)
* English | [Русский](/README_RU.md)

## Description:

A simple and convenient interface for using of various neural network models. You can create datasets, finetune, evaluate and generate with LLM and StableDiffusion, using various hyperparameters. You can also view files from the outputs directory in gallery, download the LLM and StableDiffusion models, change the application settings inside the interface and check system sensors

The goal of the project - to create the easiest possible application to finetune, evaluate and generate of neural network models

### LLM: ![1](https://github.com/Dartvauder/NeuroTrainerWebUI/assets/140557322/a1de823d-1818-4212-a821-e9b92367ffeb)

### StableDiffusion: ![2](https://github.com/Dartvauder/NeuroTrainerWebUI/assets/140557322/ed61b74c-4a2e-4f0c-ba0b-a41d600ca737)

### Gallery: ![3](https://github.com/Dartvauder/NeuroTrainerWebUI/assets/140557322/2bf3c8ec-ed37-4053-9cbe-4fef98a91a34)

### ModelDownloader: ![4](https://github.com/Dartvauder/NeuroTrainerWebUI/assets/140557322/ae2f94e0-eeb4-45b5-9450-376434a52513)

### Settings: ![5](https://github.com/Dartvauder/NeuroTrainerWebUI/assets/140557322/4c453be7-9839-41a6-9154-10447036c435)

### System: ![6](https://github.com/Dartvauder/NeuroTrainerWebUI/assets/140557322/55e01cba-dab6-4654-8d3f-2cccef4dd040)

## Features:

* Easy installation via install.bat(Windows) or install.sh(Linux)
* Flexible and optimized interface (By Gradio)
* Authentication via admin:admin (You can enter your login details in the GradioAuth.txt file)
* Support for Transformers: finetune, evaluate, quantize and generate (LLM)
* Support for Diffusers and Safetensors: finetune, evaluate, conversion and generate (StableDiffusion)
* Full and LORA types of finetune, evaluate and generate (For LLM and StableDiffusion)
* Ability to create a dataset (For LLM and StableDiffusion)
* Gallery
* ModelDownloader (For LLM and StableDiffusion)
* Application settings
* Ability to see system sensors

## Required Dependencies:

* [Python](https://www.python.org/downloads/) (3.10+)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.X) and [cuDNN](https://developer.nvidia.com/cudnn-downloads) (9.X)
* [Cmake](https://cmake.org/download/)
- C+ compiler
  - Windows: [VisualStudio](https://visualstudio.microsoft.com/ru/)
  - Linux: [GCC](https://gcc.gnu.org/)

## Minimum System Requirements:

* System: Windows or Linux
* GPU: 8GB+ or CPU: 16 core 3.6Ghz
* RAM: 24GB+
* Disk space: 10GB+
* Internet for installing

## How to install:

### Windows

1) `git clone https://github.com/Dartvauder/NeuroTrainerWebUI.git` to any location
2) Run the `Install.bat` and wait for installation
3) After installation, run `Start.bat`
4) Select the file version and wait for the application to launch
5) Now you can start experiment with your models!

To get update, run `Update.bat`
To work with the virtual environment through the terminal, run `Venv.bat`

### Linux

1) `git clone https://github.com/Dartvauder/NeuroTrainerWebUI.git` to any location
2) In the terminal, run the `./Install.sh` and wait for installation of all dependencies
3) After installation, run `./Start.sh`
4) Wait for the application to launch
5) Now you can start experiment with your models!

To get update, run `./Update.sh`
To work with the virtual environment through the terminal, run `./Venv.sh`

## How to use:

#### Interface has six tabs: LLM, StableDiffusion, Gallery, ModelDownloader, Settings and System. Select the one you need and follow the instructions below 

### LLM - has five sub-tabs:

#### Dataset:

* Here you can create a new or expand an existing dataset
* Datasets are saved in a folder *datasets/llm*

#### Finetune:

1) First upload your models to the folder: *models/llm*
2) Upload your dataset to the folder: *datasets/llm*
3) Select your model and dataset from the drop-down lists
4) Select a finetune method
5) Write a name for the model
6) Set up the model hyper-parameters for finetuning
7) Click the `Submit` button to receive the finetuned model

#### Evaluate:

1) First upload your models to the folder: *finetuned-models/llm*
2) Upload your dataset to the folder: *datasets/llm*
3) Select your models and dataset from the drop-down lists
4) Set up the models parameters for evaluate
5) Click the `Submit` button to receive the evaluate of model

#### Quantize:

1) First upload your models to the folder: *finetuned-models/llm*
2) Select a Model and Quantization Type
3) Click the `Submit` button to receive the conversion of model

#### Generate:

1) Select your models from the drop-down list
2) Set up the models according to the parameters you need
3) Set up the models parameters to generate
4) Click the `Submit` button to receive the generated text

### StableDiffusion - has five sub-tabs:

#### Dataset:

* Here you can create a new or expand an existing dataset
* Datasets are saved in a folder *datasets/sd*

#### Finetune:

1) First upload your models to the folder: *models/sd*
2) Upload your dataset to the folder: *datasets/sd*
3) Select your model and dataset from the drop-down lists
4) Select a model type and finetune method
5) Write a name for the model
6) Set up the model hyper-parameters for finetuning
7) Click the `Submit` button to receive the finetuned model

#### Evaluate:

1) First upload your models to the folder: *finetuned-models/sd*
2) Upload your dataset to the folder: *datasets/sd*
3) Select your models and dataset from the drop-down lists
4) Select a model method and model type
5) Enter your prompt
6) Set up the models parameters for evaluate
7) Click the `Submit` button to receive the evaluate of model

#### Conversion:

1) First upload your models to the folder: *finetuned-models/sd*
2) Select a model type
3) Set up the models parameters for convert
4) Click the `Submit` button to receive the conversion of model

#### Generate:

1) First upload your models to the folder: *finetuned-models/sd*
2) Select your models from the drop-down list
3) Select a model method and model type
5) Enter your prompt
6) Set up the models parameters to generate
7) Click the `Submit` button to receive the generated image

### Gallery:

* Here you can view files from the outputs directory

### ModelDownloader:

* Here you can download `LLM` and `StableDiffusion` models. Just choose the model from the drop-down list and click the `Submit` button
#### `LLM` models are downloaded here: *models/llm*
#### `StableDiffusion` models are downloaded here: *models/sd*

### Settings: 

* Here you can change the application settings. For now you can only change `Share` mode to `True` or `False`

### System: 

* Here you can see the indicators of your computer's sensors by clicking on the `Submit` button

### Additional Information:

1) All finetunes are saved in the *finetuned-models* folder
2) You can press the `Clear` button to reset your selection
3) You can turn off the application using the `Close terminal` button
4) You can open the *finetuned-models*, *datasets*, and *outputs* folders by clicking on the folder name button

## Where can i get models and datasets?

* LLM and StableDiffusion models can be taken from [HuggingFace](https://huggingface.co/models) or from ModelDownloader inside interface
* LLM and StableDiffusion datasets can be taken from [HuggingFace](https://huggingface.co/datasets) or you can create own datasets inside interface

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
* [TrainerScripts](http://www.apache.org/licenses/LICENSE-2.0)
* [CLIP](https://huggingface.co/openai/clip-vit-base-patch16)
* [BERT](https://huggingface.co/google-bert/bert-base-uncased)
* [LLAMA.CPP](https://github.com/ggerganov/llama.cpp/blob/master/LICENSE)

## Donation

### *If you liked my project and want to donate, here is options to donate. Thank you very much in advance!*

* CryptoWallet(BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)
