## [Features](/#Features) | [Dependencies](/#Required-Dependencies) | [SystemRequirements](/#Minimum-System-Requirements) | [Install](/#How-to-install) | [Usage](/#How-to-use) | [Models](/#Where-can-I-get-models) | [Wiki](/#Wiki) | [Acknowledgment](/#Acknowledgment-to-developers) | [Licenses](/#Third-Party-Licenses)

* Work in progress! (ALPHA)
* English | [Русский](/README_RU.md)

## Description:

A simple and convenient interface for using of various neural network models. You can finetune, evaluate and generate with LLM and StableDiffusion, using various hyperparameters. You can also check system sensors

The goal of the project - to create the easiest possible application to finetune, evaluate and generate with neural network models

### LLM: ![1](https://github.com/Dartvauder/NeuroTrainerWebUI/assets/140557322/80b62b35-38c5-46cb-824a-a05513cb2c4a)

### StableDiffusion: ![2](https://github.com/Dartvauder/NeuroTrainerWebUI/assets/140557322/2b00cbf9-1569-4c9e-afa7-68679edd6a79)

### System: ![3](https://github.com/Dartvauder/NeuroTrainerWebUI/assets/140557322/1382c12a-f2af-4d2c-ab4f-6e9345b107cc)

## Features:

* Easy installation via install.bat(Windows) or install.sh(Linux)
* Flexible and optimized interface (By Gradio)
* Authentication via admin:admin (You can enter your login details in the GradioAuth.txt file)
* Support for Transformers finetune, evaluate and generate (LLM)
* Support for Diffusers finetune, evaluate and generate (StableDiffusion)
* Full and LORA types of finetune, evaluate and generate (For LLM and StableDiffusion)
* Ability to see system sensors

## Required Dependencies:

* [Python](https://www.python.org/downloads/) (3.10+)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.X) and [cuDNN](https://developer.nvidia.com/cudnn-downloads) (9.X)

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
4) Select a finetune type
5) Set up the model hyper-parameters for finetuning
6) Click the `Submit` button to receive the finetuned model

#### Evaluate:

1) First upload your models to the folder: *finetuned-models/llm*
2) Upload your dataset to the folder: *datasets/llm*
3) Select your models and dataset from the drop-down lists
4) Set up the models parameters for evaluate
5) Click the `Submit` button to receive the evaluate of model

#### Generate:

1) Select your models from the drop-down list
2) Set up the models according to the parameters you need
3) Set up the models parameters to generate
4) Click the `Submit` button to receive the generated text

### StableDiffusion - has three sub-tabs:

#### Finetune:

1) First upload your models to the folder: *models/sd*
2) Upload your dataset to the folder: *datasets/sd*
3) Select your model and dataset from the drop-down lists
4) Select a finetune type
5) Set up the model hyper-parameters for finetuning
6) Click the `Submit` button to receive the finetuned model

#### Evaluate:

1) First upload your models to the folder: *finetuned-models/sd*
2) Upload your dataset to the folder: *datasets/sd*
3) Select your models and dataset from the drop-down lists
4) Set up the models parameters for evaluate
5) Click the `Submit` button to receive the evaluate of model

#### Generate:

1) Select your models from the drop-down list
2) Set up the models according to the parameters you need
3) Set up the models parameters to generate
4) Click the `Submit` button to receive the generated image

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
