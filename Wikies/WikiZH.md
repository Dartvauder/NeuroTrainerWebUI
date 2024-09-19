# 使用说明

## 如何使用：

#### 界面有六个标签：LLM、StableDiffusion、Wiki、ModelDownloader、Settings和System。选择您需要的标签并按照以下说明操作

### LLM - 有五个子标签：

#### 数据集：

* 在这里，您可以创建新的或扩展现有的数据集
* 数据集保存在 *datasets/llm* 文件夹中

#### 微调：

1) 首先将您的模型上传到文件夹：*models/llm*
2) 将您的数据集上传到文件夹：*datasets/llm*
3) 从下拉列表中选择您的模型和数据集
4) 选择微调方法
5) 为模型写一个名称
6) 设置模型微调的超参数
7) 点击`Submit`按钮以获取微调后的模型

#### 评估：

1) 首先将您的模型上传到文件夹：*finetuned-models/llm*
2) 将您的数据集上传到文件夹：*datasets/llm*
3) 从下拉列表中选择您的模型和数据集
4) 设置模型评估参数
5) 点击`Submit`按钮以获取模型评估结果

#### 量化：

1) 首先将您的模型上传到文件夹：*finetuned-models/llm*
2) 选择模型和量化类型
3) 点击`Submit`按钮以获取模型转换结果

#### 生成：

1) 从下拉列表中选择您的模型
2) 根据您需要的参数设置模型
3) 设置生成的模型参数
4) 点击`Submit`按钮以获取生成的文本

### StableDiffusion - 有五个子标签：

#### 数据集：

* 在这里，您可以创建新的或扩展现有的数据集
* 数据集保存在 *datasets/sd* 文件夹中

#### 微调：

1) 首先将您的模型上传到文件夹：*models/sd*
2) 将您的数据集上传到文件夹：*datasets/sd*
3) 从下拉列表中选择您的模型和数据集
4) 选择模型类型和微调方法
5) 为模型写一个名称
6) 设置模型微调的超参数
7) 点击`Submit`按钮以获取微调后的模型

#### 评估：

1) 首先将您的模型上传到文件夹：*finetuned-models/sd*
2) 将您的数据集上传到文件夹：*datasets/sd*
3) 从下拉列表中选择您的模型和数据集
4) 选择模型方法和模型类型
5) 输入您的提示
6) 设置模型评估参数
7) 点击`Submit`按钮以获取模型评估结果

#### 转换：

1) 首先将您的模型上传到文件夹：*finetuned-models/sd*
2) 选择模型类型
3) 设置模型转换参数
4) 点击`Submit`按钮以获取模型转换结果

#### 生成：

1) 首先将您的模型上传到文件夹：*finetuned-models/sd*
2) 从下拉列表中选择您的模型
3) 选择模型方法和模型类型
4) 输入您的提示
5) 设置生成的模型参数
6) 点击`Submit`按钮以获取生成的图像

### Wiki：

* 在这里，您可以查看项目的在线或离线维基

### ModelDownloader：

* 在这里，您可以下载`LLM`和`StableDiffusion`模型。只需从下拉列表中选择模型并点击`Submit`按钮
#### `LLM`模型下载到这里：*models/llm*
#### `StableDiffusion`模型下载到这里：*models/sd*

### Settings：

* 在这里，您可以更改应用程序设置

### System：

* 在这里，您可以通过点击`Submit`按钮查看计算机传感器的指标

### 附加信息：

1) 所有微调都保存在 *finetuned-models* 文件夹中
2) 您可以按`Clear`按钮重置您的选择
3) 您可以使用`Close terminal`按钮关闭应用程序
4) 您可以通过点击文件夹名称按钮打开 *finetuned-models*、*datasets* 和 *outputs* 文件夹
5) 您可以通过点击`Reload interface`重新加载界面下拉列表

## 我在哪里可以获得模型和数据集？

* LLM和StableDiffusion模型可以从[HuggingFace](https://huggingface.co/models)获取，或者从界面内的ModelDownloader获取
* LLM和StableDiffusion数据集可以从[HuggingFace](https://huggingface.co/datasets)获取，或者您可以在界面内创建自己的数据集
