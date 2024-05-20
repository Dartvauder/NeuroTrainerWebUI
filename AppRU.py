import os
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from diffusers import StableDiffusionPipeline, DDPMScheduler
from datasets import load_dataset
import matplotlib.pyplot as plt
import torch
import psutil
import GPUtil
from cpuinfo import get_cpu_info
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetTemperature, NVML_TEMPERATURE_GPU
import sacrebleu
from rouge import Rouge
import subprocess
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from mauve import compute_mauve
from sklearn.metrics import accuracy_score, precision_score


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


def get_system_info():
    gpu = GPUtil.getGPUs()[0]
    gpu_total_memory = f"{gpu.memoryTotal} MB"
    gpu_used_memory = f"{gpu.memoryUsed} MB"
    gpu_free_memory = f"{gpu.memoryFree} MB"

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    gpu_temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)

    cpu_info = get_cpu_info()
    cpu_temp = cpu_info.get("cpu_temp", None)

    ram = psutil.virtual_memory()
    ram_total = f"{ram.total // (1024 ** 3)} GB"
    ram_used = f"{ram.used // (1024 ** 3)} GB"
    ram_free = f"{ram.available // (1024 ** 3)} GB"

    return gpu_total_memory, gpu_used_memory, gpu_free_memory, gpu_temp, cpu_temp, ram_total, ram_used, ram_free


def get_available_llm_models():
    models_dir = "models/llm"
    os.makedirs(models_dir, exist_ok=True)

    llm_available_models = []
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if os.path.isdir(model_path):
            llm_available_models.append(model_name)

    return llm_available_models


def get_available_finetuned_llm_models():
    models_dir = "finetuned-models/llm"
    os.makedirs(models_dir, exist_ok=True)

    finetuned_available_models = []
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if os.path.isdir(model_path):
            finetuned_available_models.append(model_name)

    return finetuned_available_models


def get_available_llm_datasets():
    datasets_dir = "datasets/llm"
    os.makedirs(datasets_dir, exist_ok=True)

    llm_available_datasets = []
    for dataset_file in os.listdir(datasets_dir):
        if dataset_file.endswith(".json"):
            llm_available_datasets.append(dataset_file)

    return llm_available_datasets


def get_available_sd_models():
    models_dir = "models/sd"
    os.makedirs(models_dir, exist_ok=True)

    sd_available_models = []
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if os.path.isdir(model_path):
            sd_available_models.append(model_name)

    return sd_available_models


def get_available_finetuned_sd_models():
    models_dir = "finetuned-models/sd"
    os.makedirs(models_dir, exist_ok=True)

    finetuned_sd_available_models = []
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if os.path.isdir(model_path):
            finetuned_sd_available_models.append(model_name)

    return finetuned_sd_available_models


def get_available_sd_datasets():
    datasets_dir = "datasets/sd"
    os.makedirs(datasets_dir, exist_ok=True)

    sd_available_datasets = []
    for dataset_dir in os.listdir(datasets_dir):
        dataset_path = os.path.join(datasets_dir, dataset_dir)
        if os.path.isdir(dataset_path):
            sd_available_datasets.append(dataset_dir)

    return sd_available_datasets


def load_model_and_tokenizer(model_name, finetuned=False):
    if finetuned:
        model_path = os.path.join("finetuned-models/llm", model_name)
    else:
        model_path = os.path.join("models/llm", model_name)
    try:
        device = "cuda"
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model and tokenizer: {e}")
        return None, None


def finetune_llm(model_name, dataset_file, epochs, batch_size, learning_rate, weight_decay, warmup_steps, block_size, grad_accum_steps):
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
    plot_path = os.path.join(save_dir, model_name, f"{model_name}_loss_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return f"Fine-tuning completed. Model saved at: {save_path}", fig


def plot_llm_evaluation_metrics(metrics):
    if metrics is None:
        return None

    metrics_to_plot = ['bleu', 'rouge-1', 'rouge-2', 'rouge-l', 'mauve', 'accuracy', 'precision']
    metric_values = [metrics.get(metric, 0) for metric in metrics_to_plot]

    fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.6
    x = range(len(metrics_to_plot))
    bars = ax.bar(x, metric_values, width=bar_width, align='center', color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title('Evaluation Metrics')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

    fig.tight_layout()
    return fig


def evaluate_llm(model_name, dataset_file):
    model_path = os.path.join("finetuned-models/llm", model_name)
    model, tokenizer = load_model_and_tokenizer(model_name, finetuned=True)
    if model is None or tokenizer is None:
        return "Error loading model and tokenizer. Please check the model path.", None

    dataset_path = os.path.join("datasets/llm", dataset_file)
    try:
        eval_dataset = load_dataset('json', data_files=dataset_path)
        eval_dataset = eval_dataset['train']

        def process_examples(examples):
            input_texts = examples['input'] if 'input' in examples else [''] * len(examples['instruction'])
            instruction_texts = examples['instruction']
            output_texts = examples['output']

            texts = [f"{input_text}<sep>{instruction_text}<sep>{output_text}" for
                     input_text, instruction_text, output_text in zip(input_texts, instruction_texts, output_texts)]
            return {'input_ids': tokenizer(texts, truncation=True, padding='max_length', max_length=128)['input_ids'],
                    'attention_mask': tokenizer(texts, truncation=True, padding='max_length', max_length=128)['attention_mask'],
                    'labels': output_texts}

        eval_dataset = eval_dataset.map(process_examples, batched=True,
                                        remove_columns=['input', 'instruction', 'output'])
        eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return f"Error loading dataset. Please check the dataset path and format. Error: {e}", None

    try:
        references = eval_dataset['labels']
        predictions = [generate_text(model_name, tokenizer.decode(example['input_ids'], skip_special_tokens=True), max_length=128, temperature=0.7, top_p=0.9, top_k=20) for example in eval_dataset]

        bleu_score = sacrebleu.corpus_bleu(predictions, [references]).score

        rouge = Rouge()
        rouge_scores = rouge.get_scores(predictions, references, avg=True)

        max_length = max(len(tokenizer.encode(ref)) for ref in references)

        tokenized_references = tokenizer(references, return_tensors='pt', padding=True, truncation=True,
                                         max_length=max_length)
        tokenized_predictions = tokenizer(predictions, return_tensors='pt', padding=True, truncation=True,
                                          max_length=max_length)

        mauve_result = compute_mauve(tokenized_predictions.input_ids, tokenized_references.input_ids)
        mauve_score = mauve_result.mauve

        binary_predictions = [1 if pred else 0 for pred in predictions]
        binary_references = [1 if ref else 0 for ref in references]
        accuracy = accuracy_score(binary_references, binary_predictions)
        precision = precision_score(binary_references, binary_predictions)

        extracted_metrics = {
            'bleu': bleu_score,
            'rouge-1': rouge_scores['rouge-1']['f'],
            'rouge-2': rouge_scores['rouge-2']['f'],
            'rouge-l': rouge_scores['rouge-l']['f'],
            'mauve': mauve_score,
            'accuracy': accuracy,
            'precision': precision
        }

        fig = plot_llm_evaluation_metrics(extracted_metrics)

        plot_path = os.path.join(model_path, f"{model_name}_evaluation_plot.png")
        fig.savefig(plot_path)

        return f"Evaluation completed successfully. Results saved to {plot_path}", fig
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return f"Evaluation failed. Error: {e}", None


def generate_text(model_name, prompt, max_length, temperature, top_p, top_k):
    model, tokenizer = load_model_and_tokenizer(model_name, finetuned=True)
    if model is None or tokenizer is None:
        return "Error loading model and tokenizer. Please check the model path."

    try:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        pad_token_id = tokenizer.eos_token_id

        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

        output = model.generate(
            input_ids,
            do_sample=True,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            max_new_tokens=max_length,
            num_return_sequences=1,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=1.1,
            num_beams=5,
            no_repeat_ngram_size=2,
        )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        print(f"Error during text generation: {e}")
        return f"Text generation failed. Error: {e}"


def finetune_sd(model_name, dataset_name, instance_prompt, resolution, train_batch_size, gradient_accumulation_steps,
                learning_rate, lr_scheduler, lr_warmup_steps, max_train_steps):
    model_path = os.path.join("models/sd", model_name)
    dataset_path = os.path.join("datasets/sd", dataset_name)

    dataset = load_dataset("imagefolder", data_dir=dataset_path)

    args = [
        "accelerate", "launch", "trainer-scripts/train_dreambooth.py",
        f"--pretrained_model_name_or_path={model_path}",
        f"--instance_data_dir={dataset_path}",
        f"--output_dir=finetuned-models/sd/{model_name}",
        f"--instance_prompt={instance_prompt}",
        f"--resolution={resolution}",
        f"--train_batch_size={train_batch_size}",
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        f"--learning_rate={learning_rate}",
        f"--lr_scheduler={lr_scheduler}",
        f"--lr_warmup_steps={lr_warmup_steps}",
        f"--max_train_steps={max_train_steps}"
    ]

    subprocess.run(args)

    model_path = os.path.join("finetuned-models/sd", model_name)

    logs_dir = os.path.join(model_path, "logs", "dreambooth")
    events_files = [f for f in os.listdir(logs_dir) if f.startswith("events.out.tfevents")]
    latest_event_file = sorted(events_files)[-1]
    event_file_path = os.path.join(logs_dir, latest_event_file)

    event_acc = EventAccumulator(event_file_path)
    event_acc.Reload()

    loss_values = [s.value for s in event_acc.Scalars("loss")]
    steps = [s.step for s in event_acc.Scalars("loss")]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(steps, loss_values, marker='o', markersize=4, linestyle='-', linewidth=1)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Step')
    ax.set_title('Training Loss')
    ax.grid(True)

    plot_path = os.path.join(model_path, f"{model_name}_loss_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return f"Fine-tuning completed. Model saved at: {model_path}", fig


def evaluate_sd(model_name, dataset_name):
    model_path = os.path.join("finetuned-models/sd", model_name)
    dataset_path = os.path.join("datasets/sd", dataset_name)

    metrics = calculate_metrics(
        input1=dataset_path,
        input2=model_path,
        cuda=True,
        isc=True,
        fid=True,
        verbose=False
    )

    fid_score = metrics['frechet_inception_distance']
    inception_score = metrics['inception_score_mean']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(["FID Score", "Inception Score"], [fid_score, inception_score])
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics")
    ax.grid(True)

    plot_path = os.path.join(model_path, f"{model_name}_evaluation_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return f"Evaluation completed. FID Score: {fid_score:.2f}, Inception Score: {inception_score:.2f}", fig


def generate_image(model_name, prompt, negative_prompt, num_inference_steps, cfg_scale, width, height):
    model_path = os.path.join("finetuned-models/sd", model_name)

    model = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
    model.scheduler = DDPMScheduler.from_config(model.scheduler.config)

    image = model(prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps,
                  guidance_scale=cfg_scale, width=width, height=height).images[0]

    return image


def close_terminal():
    os._exit(1)


def open_finetuned_folder():
    outputs_folder = "finetuned-models"
    if os.path.exists(outputs_folder):
        if os.name == "nt":
            os.startfile(outputs_folder)
        else:
            os.system(f'open "{outputs_folder}"' if os.name == "darwin" else f'xdg-open "{outputs_folder}"')


llm_finetune_interface = gr.Interface(
    fn=finetune_llm,
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
        gr.Textbox(label="Fine-tuning Status", type="text"),
        gr.Plot(label="Fine-tuning Loss")
    ],
    title="NeuroTrainerWebUI (ALPHA) - LLM-Finetune",
    description="Fine-tune LLM models on a custom dataset",
    allow_flagging="never",
)

llm_evaluate_interface = gr.Interface(
    fn=evaluate_llm,
    inputs=[
        gr.Dropdown(choices=get_available_finetuned_llm_models(), label="Model"),
        gr.Dropdown(choices=get_available_llm_datasets(), label="Dataset"),
    ],
    outputs=[
        gr.Textbox(label="Evaluation Status"),
        gr.Plot(label="Evaluation Metrics"),
    ],
    title="NeuroTrainerWebUI (ALPHA) - LLM-Evaluate",
    description="Evaluate LLM models on a custom dataset",
    allow_flagging="never",
)

llm_generate_interface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Dropdown(choices=get_available_finetuned_llm_models(), label="Model"),
        gr.Textbox(label="Prompt", type="text"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max length"),
        gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, label="Top P"),
        gr.Slider(minimum=0, maximum=100, value=20, step=1, label="Top K"),
    ],
    outputs=gr.Textbox(label="Generated text", type="text"),
    title="NeuroTrainerWebUI (ALPHA) - LLM-Generate",
    description="Generate text using LLM models",
    allow_flagging="never",
)

sd_finetune_interface = gr.Interface(
    fn=finetune_sd,
    inputs=[
        gr.Dropdown(choices=get_available_sd_models(), label="Model"),
        gr.Dropdown(choices=get_available_sd_datasets(), label="Dataset"),
        gr.Textbox(label="Instance Prompt", type="text"),
        gr.Number(value=512, label="Resolution"),
        gr.Number(value=1, label="Train Batch Size"),
        gr.Number(value=1, label="Gradient Accumulation Steps"),
        gr.Number(value=5e-6, label="Learning Rate"),
        gr.Textbox(value="constant", label="LR Scheduler"),
        gr.Number(value=0, label="LR Warmup Steps"),
        gr.Number(value=400, label="Max Train Steps"),
    ],
    outputs=[
        gr.Textbox(label="Fine-tuning Status", type="text"),
        gr.Plot(label="Fine-tuning Loss")
    ],
    title="NeuroTrainerWebUI (ALPHA) - StableDiffusion-Finetune",
    description="Fine-tune Stable Diffusion models on a custom dataset",
    allow_flagging="never",
)

sd_evaluate_interface = gr.Interface(
    fn=evaluate_sd,
    inputs=[
        gr.Dropdown(choices=get_available_finetuned_sd_models(), label="Model"),
        gr.Dropdown(choices=get_available_sd_datasets(), label="Dataset"),
    ],
    outputs=[
        gr.Textbox(label="Evaluation Status"),
        gr.Plot(label="Evaluation Metrics"),
    ],
    title="NeuroTrainerWebUI (ALPHA) - StabledDiffusion-Evaluate",
    description="Evaluate fine-tuned Stable Diffusion models",
    allow_flagging="never",
)

sd_generate_interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Dropdown(choices=get_available_finetuned_sd_models(), label="Model"),
        gr.Textbox(label="Prompt", type="text"),
        gr.Textbox(label="Negative Prompt", type="text"),
        gr.Slider(minimum=1, maximum=150, value=30, step=1, label="Steps"),
        gr.Slider(minimum=1, maximum=30, value=8, step=0.5, label="CFG"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height"),
    ],
    outputs=gr.Image(label="Generated Image"),
    title="NeuroTrainerWebUI (ALPHA) - StableDiffusion-Generate",
    description="Generate images using fine-tuned Stable Diffusion models",
    allow_flagging="never",
)

system_interface = gr.Interface(
    fn=get_system_info,
    inputs=[],
    outputs=[
        gr.Textbox(label="GPU Total Memory"),
        gr.Textbox(label="GPU Used Memory"),
        gr.Textbox(label="GPU Free Memory"),
        gr.Textbox(label="GPU Temperature"),
        gr.Textbox(label="CPU Temperature"),
        gr.Textbox(label="RAM Total"),
        gr.Textbox(label="RAM Used"),
        gr.Textbox(label="RAM Free"),
    ],
    title="NeuroTrainerWebUI (ALPHA) - System",
    description="This interface displays system information",
    allow_flagging="never",
)

with gr.TabbedInterface([gr.TabbedInterface([llm_finetune_interface, llm_evaluate_interface, llm_generate_interface],
                        tab_names=["Finetune", "Evaluate", "Generate"]),
                        gr.TabbedInterface([sd_finetune_interface, sd_evaluate_interface, sd_generate_interface],
                        tab_names=["Finetune", "Evaluate", "Generate"]),
                        system_interface],
                        tab_names=["LLM", "StableDiffusion", "System"]) as app:
    close_button = gr.Button("Close terminal")
    close_button.click(close_terminal, [], [], queue=False)

    folder_button = gr.Button("Folder")
    folder_button.click(open_finetuned_folder, [], [], queue=False)

    github_link = gr.HTML(
        '<div style="text-align: center; margin-top: 20px;">'
        '<a href="https://github.com/Dartvauder/NeuroTrainerWebUI" target="_blank" style="color: blue; text-decoration: none; font-size: 16px;">'
        'GitHub'
        '</a>'
        '</div>'
    )

    app.launch(server_name="localhost", auth=authenticate)
