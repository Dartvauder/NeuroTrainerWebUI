# 神经网络训练Web界面

## [功能](/#功能) | [依赖](/#必需的依赖项) | [系统要求](/#最低系统要求) | [安装](/#如何安装) | [模型](/#我在哪里可以获得模型) | [Wiki](/#Wiki) | [致谢](/#对开发者的致谢) | [许可证](/#第三方许可证)

![project-image](https://github.com/user-attachments/assets/2a47ff0d-9131-4c3b-897b-46f7cb9e4ae2)

* 正在进行中！（Alpha版本）
* [English](/README.md) | [Русский](/Readmes/README_RU.md) | 漢語

## 描述：

一个简单且方便的界面，用于使用各种神经网络模型。您可以创建数据集，对LLM和StableDiffusion进行微调、评估和生成，使用各种超参数。您还可以查看wiki，下载LLM和StableDiffusion模型，在界面内更改应用程序设置并检查系统传感器。

项目目标 - 创建一个尽可能简单的应用程序来微调、评估和生成神经网络模型。

### LLM：<img width="1118" alt="1zh" src="https://github.com/user-attachments/assets/6cbe2513-8673-4e1d-94b7-6f88593ebd5b">

### StableDiffusion：<img width="1115" alt="2zh" src="https://github.com/user-attachments/assets/4efba30f-db7c-429a-8672-3b0d32f39cf9">

### 界面：<img width="1120" alt="3zh" src="https://github.com/user-attachments/assets/2ec0d5a8-d463-4547-8d68-d2d0576bbeba">

## 功能：

* 通过install.bat（Windows）或install.sh（Linux）轻松安装
* 灵活且优化的界面（由Gradio提供）
* 通过admin:admin进行身份验证（您可以在GradioAuth.txt文件中输入您的登录详细信息）
* 支持Transformers：微调、评估、量化和生成（LLM）
* 支持Diffusers和Safetensors：微调、评估、转换和生成（StableDiffusion）
* 完整和LORA类型的微调、评估和生成（适用于LLM和StableDiffusion）
* 能够创建数据集（适用于LLM和StableDiffusion）
* Wiki
* 模型下载器（适用于LLM和StableDiffusion）
* 应用程序设置
* 能够查看系统传感器

## 必需的依赖项：

* [Python](https://www.python.org/downloads/)（3.10.11）
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads)（12.4）和[cuDNN](https://developer.nvidia.com/cudnn-downloads)（9.1）
- C++编译器
  - Windows：[VisualStudio](https://visualstudio.microsoft.com/ru/)、[VisualStudioCode](https://code.visualstudio.com)和[Cmake](https://cmake.org)
  - Linux：[GCC](https://gcc.gnu.org/)、[VisualStudioCode](https://code.visualstudio.com)和[Cmake](https://cmake.org)

## 最低系统要求：

* 系统：Windows或Linux
* GPU：8GB+或CPU：16核3.6Ghz
* RAM：24GB+
* 磁盘空间：10GB+
* 安装需要互联网连接

## 如何安装：

### Windows

1) 首先安装所有[必需的依赖项](/#必需的依赖项)
2) 在任意位置执行`Git clone https://github.com/Dartvauder/NeuroTrainerWebUI.git`
3) 运行`Install.bat`并等待安装完成
4) 安装完成后，运行`Start.bat`
5) 等待应用程序启动
6) 现在您可以开始生成了！

要获取更新，请运行`Update.bat`
要通过终端使用虚拟环境，请运行`Venv.bat`

### Linux

1) 首先安装所有[必需的依赖项](/#必需的依赖项)
2) 在任意位置执行`Git clone https://github.com/Dartvauder/NeuroTrainerWebUI.git`
3) 在终端中运行`./Install.sh`并等待所有依赖项安装完成
4) 安装完成后，运行`./Start.sh`
5) 等待应用程序启动
6) 现在您可以开始生成了！

要获取更新，请运行`./Update.sh`
要通过终端使用虚拟环境，请运行`./Venv.sh`

## Wiki

* https://github.com/Dartvauder/NeuroTrainerWebUI/wiki

## 对开发者的致谢

#### 非常感谢这些项目，因为正是由于他们的应用程序/库，我才能够创建我的应用程序：

首先，我要感谢[PyCharm](https://www.jetbrains.com/pycharm/)和[GitHub](https://desktop.github.com)的开发者。在他们的应用程序的帮助下，我能够创建并分享我的代码。

* `gradio` - https://github.com/gradio-app/gradio
* `transformers` - https://github.com/huggingface/transformers
* `diffusers` - https://github.com/huggingface/diffusers

## 第三方许可证：

#### 许多模型都有自己的使用许可证。在使用之前，我建议您熟悉它们：

* [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE)
* [Diffusers](https://github.com/huggingface/diffusers/blob/main/LICENSE)
* [StableDiffusion1.5](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
* [StableDiffusion2](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
* [StableDiffusionXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
* [CLIP](https://huggingface.co/openai/clip-vit-base-patch16)
* [BERT](https://huggingface.co/google-bert/bert-base-uncased)
* [LLAMA.CPP](https://github.com/ggerganov/llama.cpp/blob/master/LICENSE)

#### 这些第三方仓库的代码也在我的项目中使用：

* [用于训练SD的Diffusers脚本](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)

## 捐赠

### *如果您喜欢我的项目并想要捐赠，这里有捐赠的选项。非常感谢您的支持！*

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)

## 星星的历史

[![Star History Chart](https://api.star-history.com/svg?repos=Dartvauder/NeuroTrainerWebUI&type=Date)](https://star-history.com/#Dartvauder/NeuroTrainerWebUI&Date)
