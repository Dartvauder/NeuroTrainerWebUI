## How to use:

#### Interface has four main tabs: LLM, StableDiffusion, StableAudio and Interface. Select the one you need and follow the instructions below 

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

### StableDiffusion - has six sub-tabs:

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

#### Quantize:

1) First upload your models to the folder: *finetuned-models/sd*
2) Select a Model and Quantization Type
3) Click the `Submit` button to receive the conversion of model

#### Generate:

1) First upload your models to the folder: *finetuned-models/sd*
2) Select your models from the drop-down list
3) Select a model method and model type
5) Enter your prompt
6) Set up the models parameters to generate
7) Click the `Submit` button to receive the generated image

### StableAudio - has three sub-tabs:

#### Dataset:

* Here you can create a new or expand an existing dataset
* Datasets are saved in a folder *datasets/audio*

#### Finetune:

1) First upload your models to the folder: *models/audio*
2) Upload your dataset to the folder: *datasets/audio*
3) Select your model and dataset from the drop-down lists
4) Write a name for the model
5) Click the `Submit` button to receive the finetuned model

#### Generate:

1) First upload your models to the folder: *finetuned-models/audio*
2) Select your models from the drop-down list
3) Enter your prompt
4) Set up the models parameters to generate
5) Click the `Submit` button to receive the generated audio

### Wiki:

* Here you can view online or offline wiki of project

### ModelDownloader:

* Here you can download `LLM`, `StableDiffusion` and `StableAudio` models. Just choose the model from the drop-down list and click the `Submit` button
#### `LLM` models are downloaded here: *models/llm*
#### `StableDiffusion` models are downloaded here: *models/sd*
#### `StableAudio` models are downloaded here: *models/audio*

### Settings: 

* Here you can change the application settings

### System: 

* Here you can see the indicators of your computer's sensors by clicking on the `Submit` button

### Additional Information:

1) All finetunes are saved in the *finetuned-models* folder
2) You can press the `Clear` button to reset your selection
3) You can turn off the application using the `Close terminal` button
4) You can open the *finetuned-models*, *datasets*, and *outputs* folders by clicking on the folder name button
5) You can reload interface dropdown lists by clicking on the `Reload interface`

## Where can i get models and datasets?

* LLM, StableDiffusion and StableAudio models can be taken from [HuggingFace](https://huggingface.co/models) or from ModelDownloader inside interface
* LLM, StableDiffusion and StableAudio datasets can be taken from [HuggingFace](https://huggingface.co/datasets) or you can create own datasets inside interface
