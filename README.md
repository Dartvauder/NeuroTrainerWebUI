## [Features](/#Features) | [Dependencies](/#Required-Dependencies) | [SystemRequirements](/#Minimum-System-Requirements) | [Install](/#How-to-install) | [Usage](/#How-to-use) | [Models](/#Where-can-I-get-models) | [Wiki](/#Wiki) | [Acknowledgment](/#Acknowledgment-to-developers) | [Licenses](/#Third-Party-Licenses)

* Work in progress! (ALPHA)
* English | [Русский](/README_RU.md)

## Description:

A simple and convenient interface for finetuning of various neural network models. You can finetune, evaluate and generate with LLM and StableDiffusion, using various hyperparameters.

The goal of the project - to create the easiest possible application to finetune neural network models

### LLM: 

### StableDiffusion: 

### System: 

## Features:

* Easy installation via install.bat(Windows) or install.sh(Linux)
* Flexible and optimized interface (By Gradio)
* Authentication via admin:admin (You can enter your login details in the GradioAuth.txt file)
* Support for Transformers finetune, evaluate and generate (LLM)
* Support for Diffusers finetune, evaluate and generate (StableDiffusion)
* Ability to see system sensors

## Required Dependencies:

* [Python](https://www.python.org/downloads/) (3.10+)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.X) and [cuDNN](https://developer.nvidia.com/cudnn-downloads) (9.X)

## Minimum System Requirements:

* System: Windows or Linux
* GPU: 6GB+
* RAM: 16GB+
* Disk space: 10GB+
* Internet for downloading models and installing

## How to install:

### Windows

1) `git clone https://github.com/Dartvauder/NeuroTrainerWebUI.git` to any location
2) Run the `Install.bat` and wait for installation
3) After installation, run `Start.bat`
4) Select the file version and wait for the application to launch
5) Now you can start generating!

To get update, run `Update.bat`
To work with the virtual environment through the terminal, run `Venv.bat`

### Linux

1) `git clone https://github.com/Dartvauder/NeuroTrainerWebUI.git` to any location
2) In the terminal, run the `./Install.sh` and wait for installation of all dependencies
3) After installation, run `./Start.sh`
4) Wait for the application to launch
5) Now you can start generating!

To get update, run `./Update.sh`
To work with the virtual environment through the terminal, run `./Venv.sh`

## How to use:

#### Interface has three tabs: LLM, StableDiffusion and System. Select the one you need and follow the instructions below 

### LLM - has three sub-tabs:

#### Finetune:

1) First upload your models to the folder: *models/llm*
2) Upload your dataset to the folder: *datasets/llm*
3) Select your model and dataset from the drop-down lists
4) Set up the model hyper-parameters for finetuning
5) Click the `Submit` button to receive the finetuned model

#### Evaluate:

1) First upload your models to the folder: *finetuned-models/llm*
2) Upload your dataset to the folder: *datasets/llm*
3) Select your model and dataset from the drop-down lists
4) Click the `Submit` button to receive the evaluate of model

#### Generate:

1) Select your model from the drop-down list
2) Set up the model according to the parameters you need
3) Click the `Submit` button to receive the generated text

### StableDiffusion - has three sub-tabs:

#### Finetune:

1) First upload your models to the folder: *models/sd*
2) Upload your dataset to the folder: *datasets/sd*
3) Select your model and dataset from the drop-down lists
4) Set up the model hyper-parameters for finetuning
5) Click the `Submit` button to receive the finetuned model

#### Evaluate:

1) First upload your models to the folder: *finetuned-models/sd*
2) Upload your dataset to the folder: *datasets/sd*
3) Select your model and dataset from the drop-down lists
4) Click the `Submit` button to receive the evaluate of model

#### Generate:

1) Select your model from the drop-down list
2) Set up the model according to the parameters you need
3) Click the `Submit` button to receive the generated image

### Additional Information:

1) All finetunes are saved in the *finetuned-models* folder
2) You can press the `Clear` button to reset your selection
3) You can turn off the application using the `Close terminal` button
4) You can open the *finetuned-models* folder by clicking on the `Folder` button

## Where can i get models and datasets?

* LLM and StableDiffusion models can be taken from [HuggingFace](https://huggingface.co/models)
* LLM and StableDiffusion datasets can be taken from [HuggingFace](https://huggingface.co/datasets)

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

## Donation

### *If you liked my project and want to donate, here is options to donate. Thank you very much in advance!*

* CryptoWallet(BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)
