import os
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
import matplotlib.pyplot as plt


def authenticate(username, password):
    try:
        with open("GradioAuth.txt", "r") as file:
            stored_credentials = file.read().strip().split(":")
            if len(stored_credentials) == 2:
                stored_username, stored_password = stored_credentials
                return username == stored_username and password == stored_password
    except FileNotFoundError:
        print("Authentication file not found.")
    except Exception as e:
        print(f"Error reading authentication file: {e}")
    return False


def get_available_llm_models():
    models_dir = "models/llm"
    os.makedirs(models_dir, exist_ok=True)

    llm_available_models = []
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if os.path.isdir(model_path):
            llm_available_models.append(model_name)

    return llm_available_models


def get_available_llm_datasets():
    datasets_dir = "datasets/llm"
    os.makedirs(datasets_dir, exist_ok=True)

    llm_available_datasets = []
    for dataset_file in os.listdir(datasets_dir):
        if dataset_file.endswith(".json"):
            llm_available_datasets.append(dataset_file)

    return llm_available_datasets


def load_model_and_tokenizer(model_name):
    model_path = os.path.join("models/llm", model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model and tokenizer: {e}")
        return None, None


def train_llm(model_name, dataset_file, epochs, batch_size, learning_rate, weight_decay, warmup_steps, block_size, grad_accum_steps):
    model, tokenizer = load_model_and_tokenizer(model_name)
    if model is None or tokenizer is None:
        return "Error loading model and tokenizer. Please check the model path.", None

    dataset_path = os.path.join("datasets/llm", dataset_file)
    try:
        train_dataset = load_dataset('json', data_files=dataset_path)
        train_dataset = train_dataset['train']

        def process_examples(examples):
            input_texts = examples['input'] if 'input' in examples else [''] * len(examples['instruction'])
            instruction_texts = examples['instruction']
            output_texts = examples['output']

            texts = [f"{input_text}<sep>{instruction_text}<sep>{output_text}" for
                     input_text, instruction_text, output_text in zip(input_texts, instruction_texts, output_texts)]
            return tokenizer(texts, truncation=True, padding='max_length', max_length=block_size)

        train_dataset = train_dataset.map(process_examples, batched=True,
                                          remove_columns=['input', 'instruction', 'output'])
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return f"Error loading dataset. Please check the dataset path and format. Error: {e}", None

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    save_dir = "finetuned-models/llm"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, model_name)

    training_args = TrainingArguments(
        output_dir=save_path,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=grad_accum_steps,
        save_steps=10_000,
        save_total_limit=2,
        logging_strategy='epoch',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    try:
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(save_path)
        print("Training completed successfully.")
    except Exception as e:
        print(f"Error during training: {e}")
        return f"Training failed. Error: {e}", None

    loss_values = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
    epochs = [log['epoch'] for log in trainer.state.log_history if 'epoch' in log]

    epochs = epochs[:len(loss_values)]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, loss_values, marker='o', markersize=4, linestyle='-', linewidth=1)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.set_title('Training Loss')

    if loss_values:
        ax.set_ylim(bottom=min(loss_values)-0.01, top=max(loss_values)+0.01)
        ax.set_xticks(epochs)
        ax.set_xticklabels([int(epoch) for epoch in epochs])
    else:
        print("No loss values found in trainer.state.log_history")

    ax.grid(True)

    plot_dir = save_dir
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f"{model_name}_loss_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return f"Training completed. Model saved at: {save_path}", fig


llm_train_interface = gr.Interface(
    fn=train_llm,
    inputs=[
        gr.Dropdown(choices=get_available_llm_models(), label="Model"),
        gr.Dropdown(choices=get_available_llm_datasets(), label="Dataset"),
        gr.Number(value=10, label="Epochs"),
        gr.Number(value=4, label="Batch size"),
        gr.Number(value=3e-5, label="Learning rate"),
        gr.Number(value=0.01, label="Weight decay"),
        gr.Number(value=100, label="Warmup steps"),
        gr.Number(value=128, label="Block size"),
        gr.Number(value=1, label="Gradient accumulation steps"),
    ],
    outputs=[
        gr.Textbox(label="Training status", type="text"),
        gr.Plot(label="Training Loss")
    ],
    title="LLM Fine-tuning",
    description="Fine-tune LLM models on a custom dataset",
    allow_flagging="never",
)


with gr.TabbedInterface([llm_train_interface], ["LLM-Train"]) as app:
    app.launch(server_name="localhost", auth=authenticate)
