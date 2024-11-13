## [功能](/#功能) | [依赖](/#必需依赖) | [系统要求](/#最低系统要求) | [安装](/#如何安装) | [模型](/#在哪里可以获得模型) | [维基](/#维基) | [致谢](/#致开发者的感谢) | [许可证](/#第三方许可证)

![项目图片](https://github.com/user-attachments/assets/2a47ff0d-9131-4c3b-897b-46f7cb9e4ae2)

* 正在开发中！（测试版）
* [English](/README.md) | [Русский](/TechnicalFiles/Readmes/README_RU.md) | 漢語

## 描述：

一个简单便捷的界面，用于使用各种神经网络模型。您可以创建数据集，对LLM、StableDiffusion和StableAudio进行微调、评估和生成，使用各种超参数。您还可以查看维基，下载LLM、StableDiffusion和StableAudio模型，在界面内更改应用程序设置并检查系统传感器。

项目目标 - 创建一个尽可能简单的应用程序，用于神经网络模型的微调、评估和生成。

### LLM: <img width="1115" alt="1zh" src="https://github.com/user-attachments/assets/f0102207-d2f4-46ad-841d-1e1b9ec3a50e">

### StableDiffusion: <img width="1115" alt="2zh" src="https://github.com/user-attachments/assets/9e6a1a28-b2ac-4348-8b94-4a25d83a7f10">

### StableAudio: <img width="1115" alt="3zh" src="https://github.com/user-attachments/assets/d0501eb9-47d0-45e7-af09-4f29e5492066">

### 界面: <img width="1114" alt="4zh" src="https://github.com/user-attachments/assets/2d3b1f0c-a86a-4cee-99b2-2aa6b41cd3be">

## 功能：

* 通过install.bat（Windows）或install.sh（Linux）轻松安装
* 灵活优化的界面（由Gradio提供）
* 支持Transformers：微调、评估、量化和生成（LLM）
* 支持Diffusers和Safetensors：微调、评估、转换、量化和生成（StableDiffusion）
* 支持StableAudio：微调和生成
* 完整和LORA类型的微调、评估和生成（适用于LLM和StableDiffusion）
* 创建数据集的能力（适用于LLM、StableDiffusion和StableAudio模型）
* 维基
* 模型下载器（适用于LLM和StableDiffusion）
* 应用程序设置
* 查看系统传感器的能力

## 必需依赖：

* [Python](https://www.python.org/downloads/) (3.10.11)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.4) 和 [cuDNN](https://developer.nvidia.com/cudnn-downloads) (9.1)
- C++编译器
  - Windows: [VisualStudio](https://visualstudio.microsoft.com/ru/), [VisualStudioCode](https://code.visualstudio.com) 和 [Cmake](https://cmake.org)
  - Linux: [GCC](https://gcc.gnu.org/), [VisualStudioCode](https://code.visualstudio.com) 和 [Cmake](https://cmake.org)

## 最低系统要求：

* 系统：Windows 或 Linux
* GPU: 8GB+ 或 CPU: 16核 3.6Ghz
* 内存：24GB+
* 磁盘空间：10GB+
* 安装需要互联网连接

## 如何安装：

### Windows

1) 首先安装所有[必需依赖](/#必需依赖)
2) 在任意位置执行 `Git clone https://github.com/Dartvauder/NeuroTrainerWebUI.git`
3) 运行 `Install.bat` 并等待安装完成
4) 安装完成后，运行 `Start.bat`
5) 等待应用程序启动
6) 现在您可以开始生成了！

要获取更新，运行 `Update.bat`
要通过终端使用虚拟环境，运行 `Venv.bat`

### Linux

1) 首先安装所有[必需依赖](/#必需依赖)
2) 在任意位置执行 `Git clone https://github.com/Dartvauder/NeuroTrainerWebUI.git`
3) 在终端中，运行 `./Install.sh` 并等待所有依赖安装完成
4) 安装完成后，运行 `./Start.sh`
5) 等待应用程序启动
6) 现在您可以开始生成了！

要获取更新，运行 `./Update.sh`
要通过终端使用虚拟环境，运行 `./Venv.sh`

## 维基

* https://github.com/Dartvauder/NeuroTrainerWebUI/wiki

## 致开发者的感谢

#### 非常感谢这些项目，因为正是由于他们的应用程序/库，我才能够创建我的应用程序：

首先，我要感谢 [PyCharm](https://www.jetbrains.com/pycharm/) 和 [GitHub](https://desktop.github.com) 的开发者。借助他们的应用程序，我能够创建并分享我的代码。

* `gradio` - https://github.com/gradio-app/gradio
* `transformers` - https://github.com/huggingface/transformers
* `diffusers` - https://github.com/huggingface/diffusers
* `llama.cpp-python` - https://github.com/abetlen/llama-cpp-python
* `stable-diffusion-cpp-python` - https://github.com/william-murray1204/stable-diffusion-cpp-python

## 第三方许可证：

#### 许多模型有自己的使用许可证。在使用之前，我建议您熟悉它们：

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

#### 这些第三方代码仓库也在我的项目中使用：

* [用于训练SD的Diffusers脚本](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)
* [用于训练StableAudio的StableAudioTools](https://github.com/Stability-AI/stable-audio-tools)

## 捐赠

### *如果您喜欢我的项目并想要捐赠，这里有捐赠选项。提前非常感谢！*

* [!["给我买杯咖啡"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)

## Star历史

[![Star历史图表](https://api.star-history.com/svg?repos=Dartvauder/NeuroTrainerWebUI&type=Date)](https://star-history.com/#Dartvauder/NeuroTrainerWebUI&Date)
