import os
from git import Repo
import requests
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from llama_cpp import Llama
from peft import LoraConfig, get_peft_model, PeftModel
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DDPMScheduler
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from bert_score import score
import json
from datetime import datetime
from torchmetrics.text.chrf import CHRFScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.vif import VisualInformationFidelity
from torchvision.transforms import Resize
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.scc import SpatialCorrelationCoefficient
from torchmetrics.image import SpectralDistortionIndex
from torchmetrics.image import SpectralAngleMapper
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
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


def get_available_llm_lora_models():
    models_dir = "finetuned-models/llm/lora"
    os.makedirs(models_dir, exist_ok=True)

    llm_lora_available_models = []
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if os.path.isdir(model_path):
            llm_lora_available_models.append(model_name)

    return llm_lora_available_models


def get_available_finetuned_llm_models():
    models_dir = "finetuned-models/llm/full"
    os.makedirs(models_dir, exist_ok=True)

    finetuned_available_llm_models = []
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if os.path.isdir(model_path):
            finetuned_available_llm_models.append(model_name)

    return finetuned_available_llm_models


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


def get_available_sd_lora_models():
    models_dir = "finetuned-models/sd/lora"
    os.makedirs(models_dir, exist_ok=True)

    sd_lora_available_models = []
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if os.path.isdir(model_path):
            sd_lora_available_models.append(model_name)

    return sd_lora_available_models


def get_available_finetuned_sd_models():
    models_dir = "finetuned-models/sd/full"
    os.makedirs(models_dir, exist_ok=True)

    finetuned_sd_available_models = []
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if os.path.isdir(model_path) or model_name.endswith(".safetensors"):
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
        model_path = os.path.join("finetuned-models/llm/full", model_name)
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


def create_llm_dataset(existing_file, file_name, instruction, input_text, output_text):
    if existing_file:
        file_path = os.path.join("datasets", "llm", existing_file)
        with open(file_path, "r") as f:
            data = json.load(f)
        data.append({"instruction": instruction, "input": input_text, "output": output_text})
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        return f"New column added to the existing file: {existing_file}"
    else:
        file_path = os.path.join("datasets", "llm", f"{file_name}.json")
        data = [{"instruction": instruction, "input": input_text, "output": output_text}]
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        return f"New dataset file created: {file_name}.json"


def finetune_llm(model_name, dataset_file, finetune_method, model_output_name, epochs, batch_size, learning_rate,
                 weight_decay, warmup_steps, block_size, grad_accum_steps, lora_r, lora_alpha, lora_dropout):
    model, tokenizer = load_model_and_tokenizer(model_name)
    if model is None or tokenizer is None:
        return "Error loading model and tokenizer. Please check the model path.", None

    if not model_name:
        return "Please select the model", None

    if not dataset_file:
        return "Please select the dataset", None

    if not model_output_name:
        return "Please write the model name", None

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

    if finetune_method == "Full":
        save_dir = os.path.join("finetuned-models/llm/full", model_output_name)
    elif finetune_method == "LORA":
        save_dir = os.path.join("finetuned-models/llm/lora", model_output_name)

    os.makedirs(save_dir, exist_ok=True)

    save_path = save_dir

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

    if finetune_method == "LORA":
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, config)

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
        print("Finetuning completed successfully.")
    except Exception as e:
        print(f"Error during Finetuning: {e}")
        return f"Finetuning failed. Error: {e}", None

    loss_values = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
    epochs = [log['epoch'] for log in trainer.state.log_history if 'epoch' in log]

    epochs = epochs[:len(loss_values)]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, loss_values, marker='o', markersize=4, linestyle='-', linewidth=1)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.set_title('Training Loss')

    if loss_values:
        ax.set_ylim(bottom=min(loss_values) - 0.01, top=max(loss_values) + 0.01)
        ax.set_xticks(epochs)
        ax.set_xticklabels([int(epoch) for epoch in epochs])
    else:
        print("No loss values found in trainer.state.log_history")

    ax.grid(True)

    plot_dir = save_dir
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f"{model_output_name}_loss_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return f"Finetuning completed. Model saved at: {save_path}", fig


def plot_llm_evaluation_metrics(metrics):
    if metrics is None:
        return None

    metrics_to_plot = ['bleu', 'bert', 'rouge-1', 'rouge-2', 'rouge-l', 'mauve', 'accuracy', 'precision', 'chrf']
    metric_values = [metrics.get(metric, 0) for metric in metrics_to_plot]

    fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.6
    x = range(len(metrics_to_plot))
    bars = ax.bar(x, metric_values, width=bar_width, align='center', color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#aec7e8'])

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


def evaluate_llm(model_name, lora_model_name, dataset_file, user_input, max_length, temperature, top_p, top_k):
    model_path = os.path.join("finetuned-models/llm/full", model_name)
    model, tokenizer = load_model_and_tokenizer(model_name, finetuned=True)
    if model is None or tokenizer is None:
        return "Error loading model and tokenizer. Please check the model path.", None

    if not model_name:
        return "Please select the model", None

    if not dataset_file:
        return "Please select the dataset", None

    if lora_model_name and not model_name:
        return "Please select the original model", None

    if lora_model_name:
        lora_model_path = os.path.join("finetuned-models/llm/lora", lora_model_name)
        model = PeftModel.from_pretrained(model, lora_model_path)

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
                    'attention_mask': tokenizer(texts, truncation=True, padding='max_length', max_length=128)[
                        'attention_mask'],
                    'labels': output_texts}

        eval_dataset = eval_dataset.map(process_examples, batched=True,
                                        remove_columns=['input', 'instruction', 'output'])
        eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return f"Error loading dataset. Please check the dataset path and format. Error: {e}", None

    try:
        references = eval_dataset['labels']
        predictions = [generate_text(model_name, lora_model_name,
                                     user_input if user_input else tokenizer.decode(example['input_ids'],
                                                                                    skip_special_tokens=True),
                                     max_length, temperature, top_p, top_k, output_format='txt')[0] for example in eval_dataset]

        bert_model_name = "google-bert/bert-base-uncased"
        bert_repo_url = f"https://huggingface.co/{bert_model_name}"
        bert_repo_dir = os.path.join("trainer-scripts", bert_model_name)

        if not os.path.exists(bert_repo_dir):
            Repo.clone_from(bert_repo_url, bert_repo_dir)

        predictions = [pred.strip() for pred in predictions if pred.strip()]
        references = [ref.strip() for ref in references if ref.strip()]

        bleu_score = sacrebleu.corpus_bleu(predictions, [references]).score

        P, R, F1 = score(predictions, references, lang='en', model_type=bert_model_name, num_layers=12)
        bert_score = F1.mean().item()

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

        chrf_metric = CHRFScore()
        for reference, prediction in zip(references, predictions):
            chrf_metric.update(prediction, reference)
        chrf_score = chrf_metric.compute().item()

        extracted_metrics = {
            'bleu': bleu_score,
            'bert': bert_score,
            'rouge-1': rouge_scores['rouge-1']['f'],
            'rouge-2': rouge_scores['rouge-2']['f'],
            'rouge-l': rouge_scores['rouge-l']['f'],
            'mauve': mauve_score,
            'accuracy': accuracy,
            'precision': precision,
            'chrf': chrf_score
        }

        fig = plot_llm_evaluation_metrics(extracted_metrics)

        plot_path = os.path.join(model_path, f"{model_name}_evaluation_plot.png")
        fig.savefig(plot_path)

        return f"Evaluation completed successfully. Results saved to {plot_path}", fig
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return f"Evaluation failed. Error: {e}", None


def quantize_llm(model_name, quantization_type):
    if not model_name:
        return "Please select the model"

    model_path = os.path.join("finetuned-models/llm/full", model_name)
    llama_cpp_path = "trainer-scripts/llm/llama.cpp"

    os.makedirs(os.path.dirname(llama_cpp_path), exist_ok=True)

    if not os.path.exists(llama_cpp_path):
        llama_cpp_repo_url = "https://github.com/ggerganov/llama.cpp.git"
        Repo.clone_from(llama_cpp_repo_url, llama_cpp_path)

    try:
        os.chdir(llama_cpp_path)

        subprocess.run(f"cmake -B build", shell=True, check=True)
        subprocess.run(f"cmake --build build --config Release", shell=True, check=True)

        subprocess.run(f"python convert.py {model_path}", shell=True, check=True)

        input_model = os.path.join(model_path, "ggml-model-f16.ggml")
        output_model = os.path.join(model_path, f"ggml-model-{quantization_type}.ggml")
        subprocess.run(f"./quantize {input_model} {output_model} {quantization_type}", shell=True, check=True)

        return f"Quantization completed successfully. Model saved at: {output_model}"

    except subprocess.CalledProcessError as e:
        return f"Error during quantization: {e}"
    finally:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))


def generate_text(model_name, lora_model_name, model_type, prompt, max_length, temperature, top_p, top_k, output_format):
    if model_type == "transformers":
        model, tokenizer = load_model_and_tokenizer(model_name, finetuned=True)
        if model is None or tokenizer is None:
            return None, "Error loading model and tokenizer. Please check the model path."

        if not model_name:
            return None, "Please select the model"

        if lora_model_name and not model_name:
            return None, "Please select the original model"

        if lora_model_name:
            lora_model_path = os.path.join("finetuned-models/llm/lora", lora_model_name)
            model = PeftModel.from_pretrained(model, lora_model_path)

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

            output_dir = "outputs/llm"
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_file = f"llm_history_{timestamp}.{output_format}"
            output_path = os.path.join(output_dir, output_file)

            if output_format == "txt":
                with open(output_path, "a", encoding="utf-8") as file:
                    file.write(f"Human: {prompt}\nAI: {generated_text}\n")
            elif output_format == "json":
                history = [{"Human": prompt, "AI": generated_text}]
                with open(output_path, "w", encoding="utf-8") as file:
                    json.dump(history, file, indent=2, ensure_ascii=False)

            return generated_text, "Text generation successful"
        except Exception as e:
            print(f"Error during text generation: {e}")
            return None, f"Text generation failed. Error: {e}"

    elif model_type == "llama.cpp":
        model_path = os.path.join("finetuned-models/llm/full", model_name)

        try:

            llm = Llama(model_path=model_path, n_ctx=max_length, n_parts=-1, seed=-1, f16_kv=True, logits_all=False, vocab_only=False, use_mlock=False, n_threads=8, n_batch=1, suffix=None)

            output = llm(prompt, max_tokens=max_length, top_k=top_k, top_p=top_p, temperature=temperature, stop=None, echo=False)

            generated_text = output['choices'][0]['text']

            output_dir = "outputs/llm"
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_file = f"llm_history_{timestamp}.{output_format}"
            output_path = os.path.join(output_dir, output_file)

            if output_format == "txt":
                with open(output_path, "a", encoding="utf-8") as file:
                    file.write(f"Human: {prompt}\nAI: {generated_text}\n")
            elif output_format == "json":
                history = [{"Human": prompt, "AI": generated_text}]
                with open(output_path, "w", encoding="utf-8") as file:
                    json.dump(history, file, indent=2, ensure_ascii=False)

            return generated_text, "Text generation successful"
        except Exception as e:
            print(f"Error during text generation: {e}")
            return None, f"Text generation failed. Error: {e}"


def create_sd_dataset(image_files, existing_dataset, dataset_name, file_prefix, prompt_text):
    if existing_dataset:
        dataset_dir = os.path.join("datasets", "sd", existing_dataset, "train")
    else:
        dataset_dir = os.path.join("datasets", "sd", dataset_name, "train")

    os.makedirs(dataset_dir, exist_ok=True)

    metadata_file = os.path.join(dataset_dir, "metadata.jsonl")

    with open(metadata_file, "a") as f:
        for i, image_file in enumerate(image_files):
            file_name = f"{file_prefix}-{i + 1}.jpg"
            image_path = os.path.join(dataset_dir, file_name)
            image = Image.open(image_file.name)
            image.save(image_path)

            metadata = {
                "file_name": file_name,
                "text": prompt_text
            }
            f.write(json.dumps(metadata) + "\n")

    return f"Dataset {'updated' if existing_dataset else 'created'} successfully at {dataset_dir}"


def finetune_sd(model_name, dataset_name, model_type, finetune_method, model_output_name, resolution,
                train_batch_size, gradient_accumulation_steps,
                learning_rate, lr_scheduler, lr_warmup_steps, max_train_steps, rank):
    model_path = os.path.join("models/sd", model_name)
    dataset_path = os.path.join("datasets/sd", dataset_name)

    if not model_name:
        return "Please select the model", None

    if not dataset_name:
        return "Please select the dataset", None

    if not model_output_name:
        return "Please write the model name", None

    if finetune_method == "Full":
        output_dir = os.path.join("finetuned-models/sd/full", model_output_name)
        if model_type == "SD":
            dataset = load_dataset("imagefolder", data_dir=dataset_path)
            args = [
                "accelerate", "launch", "trainer-scripts/sd/train_text_to_image.py",
                f"--pretrained_model_name_or_path={model_path}",
                f"--train_data_dir={dataset_path}",
                f"--output_dir={output_dir}",
                f"--resolution={resolution}",
                f"--train_batch_size={train_batch_size}",
                f"--gradient_accumulation_steps={gradient_accumulation_steps}",
                f"--learning_rate={learning_rate}",
                f"--lr_scheduler={lr_scheduler}",
                f"--lr_warmup_steps={lr_warmup_steps}",
                f"--max_train_steps={max_train_steps}",
                f"--caption_column=text",
                f"--mixed_precision=no",
                f"--seed=0"
            ]
        elif model_type == "SDXL":
            dataset = load_dataset("imagefolder", data_dir=dataset_path)
            args = [
                "accelerate", "launch", "trainer-scripts/sd/train_text_to_image_sdxl.py",
                f"--pretrained_model_name_or_path={model_path}",
                f"--train_data_dir={dataset_path}",
                f"--output_dir={output_dir}",
                f"--resolution={resolution}",
                f"--train_batch_size={train_batch_size}",
                f"--gradient_accumulation_steps={gradient_accumulation_steps}",
                f"--learning_rate={learning_rate}",
                f"--lr_scheduler={lr_scheduler}",
                f"--lr_warmup_steps={lr_warmup_steps}",
                f"--max_train_steps={max_train_steps}",
                f"--caption_column=text",
                f"--mixed_precision=no",
                f"--seed=0"
            ]
    elif finetune_method == "LORA":
        output_dir = os.path.join("finetuned-models/sd/lora", model_output_name)
        if model_type == "SD":
            dataset = load_dataset("imagefolder", data_dir=dataset_path)
            args = [
                "accelerate", "launch", "trainer-scripts/sd/train_text_to_image_lora.py",
                f"--pretrained_model_name_or_path={model_path}",
                f"--train_data_dir={dataset_path}",
                f"--output_dir={output_dir}",
                f"--resolution={resolution}",
                f"--train_batch_size={train_batch_size}",
                f"--gradient_accumulation_steps={gradient_accumulation_steps}",
                f"--learning_rate={learning_rate}",
                f"--lr_scheduler={lr_scheduler}",
                f"--lr_warmup_steps={lr_warmup_steps}",
                f"--max_train_steps={max_train_steps}",
                f"--rank={rank}",
                f"--caption_column=text",
                f"--mixed_precision=no",
                f"--seed=0"
            ]
        elif model_type == "SDXL":
            dataset = load_dataset("imagefolder", data_dir=dataset_path)
            args = [
                "accelerate", "launch", "trainer-scripts/sd/train_text_to_image_lora_sdxl.py",
                f"--pretrained_model_name_or_path={model_path}",
                f"--train_data_dir={dataset_path}",
                f"--output_dir={output_dir}",
                f"--resolution={resolution}",
                f"--train_batch_size={train_batch_size}",
                f"--gradient_accumulation_steps={gradient_accumulation_steps}",
                f"--learning_rate={learning_rate}",
                f"--lr_scheduler={lr_scheduler}",
                f"--lr_warmup_steps={lr_warmup_steps}",
                f"--max_train_steps={max_train_steps}",
                f"--rank={rank}",
                f"--caption_column=text",
                f"--mixed_precision=no",
                f"--seed=0"
            ]
    else:
        raise ValueError(f"Invalid finetune method: {finetune_method}")

    subprocess.run(args)

    if finetune_method == "Full":
        logs_dir = os.path.join(output_dir, "logs", "text2image-fine-tune")
        events_files = [f for f in os.listdir(logs_dir) if f.startswith("events.out.tfevents")]
        latest_event_file = sorted(events_files)[-1]
        event_file_path = os.path.join(logs_dir, latest_event_file)

        event_acc = EventAccumulator(event_file_path)
        event_acc.Reload()

        loss_values = [s.value for s in event_acc.Scalars("train_loss")]
        steps = [s.step for s in event_acc.Scalars("train_loss")]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(steps, loss_values, marker='o', markersize=4, linestyle='-', linewidth=1)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Step')
        ax.set_title('Training Loss')
        ax.grid(True)

        plot_path = os.path.join(output_dir, f"{model_name}_loss_plot.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        return f"Fine-tuning completed. Model saved at: {output_dir}", fig

    elif finetune_method == "LORA":
        logs_dir = os.path.join(output_dir, "logs", "text2image-fine-tune")
        events_files = [f for f in os.listdir(logs_dir) if f.startswith("events.out.tfevents")]
        latest_event_file = sorted(events_files)[-1]
        event_file_path = os.path.join(logs_dir, latest_event_file)

        event_acc = EventAccumulator(event_file_path)
        event_acc.Reload()

        loss_values = [s.value for s in event_acc.Scalars("train_loss")]
        steps = [s.step for s in event_acc.Scalars("train_loss")]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(steps, loss_values, marker='o', markersize=4, linestyle='-', linewidth=1)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Step')
        ax.set_title('Training Loss')
        ax.grid(True)

        plot_path = os.path.join(output_dir, f"{model_name}_loss_plot.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        return f"Finetuning completed. Model saved at: {output_dir}", fig


def plot_sd_evaluation_metrics(metrics):
    metrics_to_plot = ["FID", "KID", "Inception Score", "VIF", "CLIP Score", "LPIPS", "SCC", "SDI", "SAM", "SSIM"]
    metric_values = [metrics[metric] for metric in metrics_to_plot]

    fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.6
    x = range(len(metrics_to_plot))
    bars = ax.bar(x, metric_values, width=bar_width, align="center", color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#bcbd22', '#17becf', '#aaffc3', '#ffe119'])

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, rotation=45, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("Evaluation Metrics")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")

    fig.tight_layout()
    return fig


def evaluate_sd(model_name, lora_model_name, dataset_name, model_method, model_type, user_prompt, negative_prompt, num_inference_steps, cfg_scale):
    if model_method == "Diffusers":
        if model_type == "SD":
            model_path = os.path.join("finetuned-models/sd/full", model_name)
            model = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16,
                                                            safety_checker=None).to(
                "cuda")
            model.scheduler = DDPMScheduler.from_config(model.scheduler.config)
        elif model_type == "SDXL":
            model_path = os.path.join("finetuned-models/sd/full", model_name)
            model = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=torch.float16, attention_slice=1,
                                                              safety_checker=None).to(
                "cuda")
            model.scheduler = DDPMScheduler.from_config(model.scheduler.config)
    elif model_method == "Safetensors":
        if model_type == "SD":
            model_path = os.path.join("finetuned-models/sd/full", model_name)
            model = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch.float16,
                                                             safety_checker=None).to(
                "cuda")
            model.scheduler = DDPMScheduler.from_config(model.scheduler.config)
        elif model_type == "SDXL":
            model_path = os.path.join("finetuned-models/sd/full", model_name)
            model = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=torch.float16, attention_slice=1,
                                                               safety_checker=None).to(
                "cuda")
            model.scheduler = DDPMScheduler.from_config(model.scheduler.config)
    else:
        return "Invalid model type selected", None

    if not model_name:
        return "Please select the model", None

    if not dataset_name:
        return "Please select the dataset", None

    if lora_model_name and not model_name:
        return "Please select the original model", None

    if lora_model_name:
        lora_model_path = os.path.join("finetuned-models/sd/lora", lora_model_name)
        model.unet.load_attn_procs(lora_model_path)

    dataset_path = os.path.join("datasets/sd", dataset_name)
    dataset = load_dataset("imagefolder", data_dir=dataset_path)

    num_samples = len(dataset["train"])
    subset_size = min(num_samples, 50)

    fid = FrechetInceptionDistance().to("cuda")
    kid = KernelInceptionDistance(subset_size=subset_size).to("cuda")
    inception = InceptionScore().to("cuda")
    vif = VisualInformationFidelity().to("cuda")
    lpips = LearnedPerceptualImagePatchSimilarity().to("cuda")
    scc = SpatialCorrelationCoefficient().to("cuda")
    sdi = SpectralDistortionIndex().to("cuda")
    sam = SpectralAngleMapper().to("cuda")
    ssim = StructuralSimilarityIndexMeasure().to("cuda")

    clip_model_name = "openai/clip-vit-base-patch16"
    clip_repo_url = f"https://huggingface.co/{clip_model_name}"
    clip_repo_dir = os.path.join("trainer-scripts", clip_model_name)

    if not os.path.exists(clip_repo_dir):
        Repo.clone_from(clip_repo_url, clip_repo_dir)

    clip_score = CLIPScore(model_name_or_path=clip_model_name).to("cuda")

    resize = Resize((512, 512))

    clip_scores = []

    for batch in dataset["train"]:
        image = batch["image"].convert("RGB")
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).to("cuda").to(torch.uint8)

        generated_images = model(prompt=user_prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, guidance_scale=cfg_scale,
                                 output_type="pil").images
        generated_image = generated_images[0].resize((image.width, image.height))
        generated_image_tensor = torch.from_numpy(np.array(generated_image)).permute(2, 0, 1).unsqueeze(0).to(
            "cuda").to(torch.uint8)

        fid.update(resize(image_tensor), real=True)
        fid.update(resize(generated_image_tensor), real=False)

        kid.update(resize(image_tensor), real=True)
        kid.update(resize(generated_image_tensor), real=False)

        inception.update(resize(generated_image_tensor))

        vif.update(resize(image_tensor).to(torch.float32), resize(generated_image_tensor).to(torch.float32))

        lpips_score = lpips(resize(image_tensor).to(torch.float32), resize(generated_image_tensor).to(torch.float32))
        scc_score = scc(resize(image_tensor).to(torch.float32), resize(generated_image_tensor).to(torch.float32))
        sdi_score = sdi(resize(image_tensor).to(torch.float32), resize(generated_image_tensor).to(torch.float32))
        sam_score = sam(resize(image_tensor).to(torch.float32), resize(generated_image_tensor).to(torch.float32))
        ssim_score = ssim(resize(image_tensor).to(torch.float32), resize(generated_image_tensor).to(torch.float32))

        clip_score_value = clip_score(resize(generated_image_tensor).to(torch.float32), "a photo of a generated image")
        clip_scores.append(clip_score_value.detach().item())

    fid_score = fid.compute()
    kid_score, _ = kid.compute()
    inception_score, _ = inception.compute()
    vif_score = vif.compute()
    clip_score_avg = np.mean(clip_scores)

    metrics = {
        "FID": fid_score.item(),
        "KID": kid_score.item(),
        "Inception Score": inception_score.item(),
        "VIF": vif_score.item(),
        "CLIP Score": clip_score_avg,
        "LPIPS": lpips_score.item(),
        "SCC": scc_score.item(),
        "SDI": sdi_score.item(),
        "SAM": sam_score.item(),
        "SSIM": ssim_score.item()
    }

    fig = plot_sd_evaluation_metrics(metrics)

    if model_method == "Diffusers":
        plot_path = os.path.join(model_path, f"{model_name}_evaluation_plot.png")
    elif model_method == "Safetensors":
        plot_path = os.path.join("finetuned-models/sd/full", f"{model_name}_evaluation_plot.png")
    fig.savefig(plot_path)

    return f"Evaluation completed successfully. Results saved to {plot_path}", fig


def convert_sd_model_to_safetensors(model_name, model_type, use_half, use_safetensors):
    model_path = os.path.join("finetuned-models/sd/full", model_name)
    output_path = os.path.join("finetuned-models/sd/full")

    if model_type == "SD":
        try:
            args = [
                "py",
                "trainer-scripts/sd/convert_diffusers_to_original_stable_diffusion.py",
                "--model_path", model_path,
                "--checkpoint_path", output_path,
            ]
            if use_half:
                args.append("--half")
            if use_safetensors:
                args.append("--use_safetensors")

            subprocess.run(args, check=True)

            return f"Model successfully converted to single file and saved to {output_path}"
        except subprocess.CalledProcessError as e:
            return f"Error converting model to single file: {e}"
    elif model_type == "SDXL":
        try:
            args = [
                "py",
                "trainer-scripts/sd/convert_diffusers_to_original_sdxl.py",
                "--model_path", model_path,
                "--checkpoint_path", output_path,
            ]
            if use_half:
                args.append("--half")
            if use_safetensors:
                args.append("--use_safetensors")

            subprocess.run(args, check=True)

            return f"Model successfully converted to single file and saved to {output_path}"
        except subprocess.CalledProcessError as e:
            return f"Error converting model to single file: {e}"


def generate_image(model_name, lora_model_name, model_method, model_type, prompt, negative_prompt, num_inference_steps, cfg_scale, width, height, output_format):
    if model_method == "Diffusers":
        if model_type == "SD":
            model_path = os.path.join("finetuned-models/sd/full", model_name)
            model = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16,
                                                            safety_checker=None).to(
                "cuda")
            model.scheduler = DDPMScheduler.from_config(model.scheduler.config)
        elif model_type == "SDXL":
            model_path = os.path.join("finetuned-models/sd/full", model_name)
            model = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=torch.float16, attention_slice=1,
                                                            safety_checker=None).to(
                "cuda")
            model.scheduler = DDPMScheduler.from_config(model.scheduler.config)
    elif model_method == "Safetensors":
        if model_type == "SD":
            model_path = os.path.join("finetuned-models/sd/full", model_name)
            model = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch.float16,
                                                            safety_checker=None).to(
                "cuda")
            model.scheduler = DDPMScheduler.from_config(model.scheduler.config)
        elif model_type == "SDXL":
            model_path = os.path.join("finetuned-models/sd/full", model_name)
            model = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=torch.float16, attention_slice=1,
                                                              safety_checker=None).to(
                "cuda")
            model.scheduler = DDPMScheduler.from_config(model.scheduler.config)
    else:
        return "Invalid model type selected", None

    if not model_name:
        return "Please select the model", None

    if lora_model_name and not model_name:
        return "Please select the original model", None

    if lora_model_name:
        lora_model_path = os.path.join("finetuned-models/sd/lora", lora_model_name)
        model.unet.load_attn_procs(lora_model_path)

    image = model(prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps,
                  guidance_scale=cfg_scale, width=width, height=height).images[0]

    output_dir = "outputs/sd"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"sd_image_{timestamp}.{output_format}"
    output_path = os.path.join(output_dir, output_file)

    image.save(output_path)

    return image, "Image generation successful"


def close_terminal():
    os._exit(1)


def open_finetuned_folder():
    outputs_folder = "finetuned-models"
    if os.path.exists(outputs_folder):
        if os.name == "nt":
            os.startfile(outputs_folder)
        else:
            os.system(f'open "{outputs_folder}"' if os.name == "darwin" else f'xdg-open "{outputs_folder}"')


def open_datasets_folder():
    outputs_folder = "datasets"
    if os.path.exists(outputs_folder):
        if os.name == "nt":
            os.startfile(outputs_folder)
        else:
            os.system(f'open "{outputs_folder}"' if os.name == "darwin" else f'xdg-open "{outputs_folder}"')


def open_outputs_folder():
    outputs_folder = "outputs"
    if os.path.exists(outputs_folder):
        if os.name == "nt":
            os.startfile(outputs_folder)
        else:
            os.system(f'open "{outputs_folder}"' if os.name == "darwin" else f'xdg-open "{outputs_folder}"')


def download_model(model_name_llm, model_name_sd):
    if not model_name_llm and not model_name_sd:
        return "Please select a model to download"

    if model_name_llm and model_name_sd:
        return "Please select one model type for downloading"

    if model_name_llm:
        model_url = ""
        if model_name_llm == "StableLM2-1_6B-Chat":
            model_url = "https://huggingface.co/stabilityai/stablelm-2-1_6b-chat"
        elif model_name_llm == "Qwen1.5-4B-Chat":
            model_url = "https://huggingface.co/Qwen/Qwen1.5-4B-Chat"
        model_path = os.path.join("models", "llm", model_name_llm)

        if model_url:
            response = requests.get(model_url, allow_redirects=True)
            with open(model_path, "wb") as file:
                file.write(response.content)
            return f"LLM model {model_name_llm} downloaded successfully!"
        else:
            return "Invalid LLM model name"

    if model_name_sd:
        model_url = ""
        if model_name_sd == "StableDiffusion1.5":
            model_url = "https://huggingface.co/runwayml/stable-diffusion-v1-5"
        elif model_name_sd == "StableDiffusionXL":
            model_url = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0"
        model_path = os.path.join("models", "sd", model_name_sd)

        if model_url:
            response = requests.get(model_url, allow_redirects=True)
            with open(model_path, "wb") as file:
                file.write(response.content)
            return f"StableDiffusion model {model_name_sd} downloaded successfully!"
        else:
            return "Invalid StableDiffusion model name"


def settings_interface(share_value):
    global share_mode
    share_mode = share_value == "True"
    message = f"Settings updated successfully!"

    app.launch(share=share_mode, server_name="localhost")

    return message


share_mode = False

llm_dataset_interface = gr.Interface(
    fn=create_llm_dataset,
    inputs=[
        gr.Dropdown(choices=get_available_llm_datasets(), label="Existing Dataset (optional)"),
        gr.Textbox(label="Dataset Name", type="text"),
        gr.Textbox(label="Instruction", type="text"),
        gr.Textbox(label="Input", type="text"),
        gr.Textbox(label="Output", type="text"),
    ],
    outputs=[
        gr.Textbox(label="Status", type="text"),
    ],
    title="NeuroTrainerWebUI (ALPHA) - LLM-Dataset",
    description="Create a new dataset or add a new column to an existing dataset for LLM",
    allow_flagging="never",
)

llm_finetune_interface = gr.Interface(
    fn=finetune_llm,
    inputs=[
        gr.Dropdown(choices=get_available_llm_models(), label="Model"),
        gr.Dropdown(choices=get_available_llm_datasets(), label="Dataset"),
        gr.Radio(choices=["Full", "LORA"], value="Full", label="Finetune Method"),
        gr.Textbox(label="Output Model Name", type="text"),
        gr.Number(value=10, label="Epochs"),
        gr.Number(value=4, label="Batch size"),
        gr.Number(value=3e-5, label="Learning rate"),
        gr.Number(value=0.01, label="Weight decay"),
        gr.Number(value=100, label="Warmup steps"),
        gr.Number(value=128, label="Block size"),
        gr.Number(value=1, label="Gradient accumulation steps"),
        gr.Number(value=16, label="LORA r"),
        gr.Number(value=32, label="LORA alpha"),
        gr.Number(value=0.05, label="LORA dropout"),
    ],
    outputs=[
        gr.Textbox(label="Finetuning Status", type="text"),
        gr.Plot(label="Finetuning Loss")
    ],
    title="NeuroTrainerWebUI (ALPHA) - LLM-Finetune",
    description="Finetune LLM models on a custom dataset",
    allow_flagging="never",
)

llm_evaluate_interface = gr.Interface(
    fn=evaluate_llm,
    inputs=[
        gr.Dropdown(choices=get_available_finetuned_llm_models(), label="Model"),
        gr.Dropdown(choices=get_available_llm_lora_models(), label="LORA Model (optional)"),
        gr.Dropdown(choices=get_available_llm_datasets(), label="Dataset"),
        gr.Textbox(label="Request", type="text"),
        gr.Slider(minimum=1, maximum=2048, value=128, step=1, label="Max Length"),
        gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, label="Top P"),
        gr.Slider(minimum=0, maximum=100, value=20, step=1, label="Top K"),
    ],
    outputs=[
        gr.Textbox(label="Evaluation Status"),
        gr.Plot(label="Evaluation Metrics"),
    ],
    title="NeuroTrainerWebUI (ALPHA) - LLM-Evaluate",
    description="Evaluate finetuned LLM models on a custom dataset",
    allow_flagging="never",
)

llm_quantize_interface = gr.Interface(
    fn=quantize_llm,
    inputs=[
        gr.Dropdown(choices=get_available_finetuned_llm_models(), label="Model"),
        gr.Radio(choices=["Q2_K_M", "Q4_K_M", "Q6_K_M", "Q8_K_M"], value="Q4_K_M", label="Quantization Type"),
    ],
    outputs=[
        gr.Textbox(label="Quantization Status", type="text"),
    ],
    title="NeuroTrainerWebUI (ALPHA) - LLM-Quantize",
    description="Quantize finetuned LLM models to .gguf format using llama.cpp",
    allow_flagging="never",
)

llm_generate_interface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Dropdown(choices=get_available_finetuned_llm_models(), label="Model"),
        gr.Dropdown(choices=get_available_llm_lora_models(), label="LORA Model (optional)"),
        gr.Radio(choices=["transformers", "llama.cpp"], value="transformers", label="Model Type"),
        gr.Textbox(label="Request", type="text"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max length"),
        gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, label="Top P"),
        gr.Slider(minimum=0, maximum=100, value=20, step=1, label="Top K"),
        gr.Radio(choices=["txt", "json"], value="txt", label="Output Format"),
    ],
    outputs=[
        gr.Textbox(label="Generated text", type="text"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroTrainerWebUI (ALPHA) - LLM-Generate",
    description="Generate text using finetuned LLM models",
    allow_flagging="never",
)

sd_dataset_interface = gr.Interface(
    fn=create_sd_dataset,
    inputs=[
        gr.Files(label="Image Files"),
        gr.Dropdown(choices=get_available_sd_datasets(), label="Existing Dataset (optional)"),
        gr.Textbox(label="Dataset Name"),
        gr.Textbox(label="Files Name"),
        gr.Textbox(label="Prompt Text")
    ],
    outputs=[
        gr.Textbox(label="Status")
    ],
    title="NeuroTrainerWebUI (ALPHA) - StableDiffusion-Dataset",
    description="Create a new dataset or add a new column to an existing dataset for Stable Diffusion",
    allow_flagging="never",
)

sd_finetune_interface = gr.Interface(
    fn=finetune_sd,
    inputs=[
        gr.Dropdown(choices=get_available_sd_models(), label="Model"),
        gr.Dropdown(choices=get_available_sd_datasets(), label="Dataset"),
        gr.Radio(choices=["SD", "SDXL"], value="SD", label="Model Type"),
        gr.Radio(choices=["Full", "LORA"], value="Full", label="Finetune Method"),
        gr.Textbox(label="Output Model Name", type="text"),
        gr.Number(value=512, label="Resolution"),
        gr.Number(value=1, label="Train Batch Size"),
        gr.Number(value=1, label="Gradient Accumulation Steps"),
        gr.Number(value=5e-6, label="Learning Rate"),
        gr.Textbox(value="constant", label="LR Scheduler"),
        gr.Number(value=0, label="LR Warmup Steps"),
        gr.Number(value=400, label="Max Train Steps"),
        gr.Number(value=4, label="LORA Rank"),
    ],
    outputs=[
        gr.Textbox(label="Finetuning Status", type="text"),
        gr.Plot(label="Finetuning Loss")
    ],
    title="NeuroTrainerWebUI (ALPHA) - StableDiffusion-Finetune",
    description="Finetune Stable Diffusion models on a custom dataset",
    allow_flagging="never",
)

sd_evaluate_interface = gr.Interface(
    fn=evaluate_sd,
    inputs=[
        gr.Dropdown(choices=get_available_finetuned_sd_models(), label="Model"),
        gr.Dropdown(choices=get_available_sd_lora_models(), label="LORA Model (optional)"),
        gr.Dropdown(choices=get_available_sd_datasets(), label="Dataset"),
        gr.Radio(choices=["Diffusers", "Safetensors"], value="Diffusers", label="Model Method"),
        gr.Radio(choices=["SD", "SDXL"], value="SD", label="Model Type"),
        gr.Textbox(label="Prompt", type="text"),
        gr.Textbox(label="Negative Prompt", type="text"),
        gr.Slider(minimum=1, maximum=150, value=30, step=1, label="Steps"),
        gr.Slider(minimum=1, maximum=30, value=8, step=0.5, label="CFG"),
    ],
    outputs=[
        gr.Textbox(label="Evaluation Status"),
        gr.Plot(label="Evaluation Metrics"),
    ],
    title="NeuroTrainerWebUI (ALPHA) - StabledDiffusion-Evaluate",
    description="Evaluate finetuned Stable Diffusion models on a custom dataset",
    allow_flagging="never",
)

sd_convert_interface = gr.Interface(
    fn=convert_sd_model_to_safetensors,
    inputs=[
        gr.Dropdown(choices=get_available_finetuned_sd_models(), label="Model"),
        gr.Radio(choices=["SD", "SDXL"], value="SD", label="Model Type"),
        gr.Checkbox(label="Use Half Precision", value=False),
        gr.Checkbox(label="Use Safetensors", value=False),
    ],
    outputs=[
        gr.Textbox(label="Conversion Status", type="text"),
    ],
    title="NeuroTrainerWebUI (ALPHA) - StableDiffusion-Conversion",
    description="Convert finetuned Stable Diffusion models to single file (.ckpt or .safetensors)",
    allow_flagging="never",
)

sd_generate_interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Dropdown(choices=get_available_finetuned_sd_models(), label="Model"),
        gr.Dropdown(choices=get_available_sd_lora_models(), label="LORA Model (optional)"),
        gr.Radio(choices=["Diffusers", "Safetensors"], value="Diffusers", label="Model Method"),
        gr.Radio(choices=["SD", "SDXL"], value="SD", label="Model Type"),
        gr.Textbox(label="Prompt", type="text"),
        gr.Textbox(label="Negative Prompt", type="text"),
        gr.Slider(minimum=1, maximum=150, value=30, step=1, label="Steps"),
        gr.Slider(minimum=1, maximum=30, value=8, step=0.5, label="CFG"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height"),
        gr.Radio(choices=["png", "jpeg"], value="png", label="Output Format"),
    ],
    outputs=[
        gr.Image(label="Generated Image"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroTrainerWebUI (ALPHA) - StableDiffusion-Generate",
    description="Generate images using finetuned Stable Diffusion models",
    allow_flagging="never",
)

model_downloader_interface = gr.Interface(
    fn=download_model,
    inputs=[
        gr.Dropdown(choices=[None, "StableLM2-1_6B-Chat", "Qwen1.5-4B-Chat"], label="Download LLM model", value=None),
        gr.Dropdown(choices=[None, "StableDiffusion1.5", "StableDiffusionXL"], label="Download StableDiffusion model", value=None),
    ],
    outputs=[
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroTrainerWebUI (ALPHA) - ModelDownloader",
    description="This interface allows you to download LLM and StableDiffusion models",
    allow_flagging="never",
)

settings_interface = gr.Interface(
    fn=settings_interface,
    inputs=[
        gr.Radio(choices=["True", "False"], label="Share Mode", value="False")
    ],
    outputs=[
        gr.Textbox(label="Message", type="text")
    ],
    title="NeuroTrainerWebUI (ALPHA) - Settings",
    description="This interface allows you to change settings of application",
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

with gr.TabbedInterface([gr.TabbedInterface([llm_dataset_interface, llm_finetune_interface, llm_evaluate_interface, llm_quantize_interface, llm_generate_interface],
                                            tab_names=["Dataset", "Finetune", "Evaluate", "Quantize", "Generate"]),
                         gr.TabbedInterface([sd_dataset_interface, sd_finetune_interface, sd_evaluate_interface, sd_convert_interface, sd_generate_interface],
                                            tab_names=["Dataset", "Finetune", "Evaluate", "Conversion", "Generate"]),
                         model_downloader_interface, settings_interface, system_interface],
                        tab_names=["LLM", "StableDiffusion", "Settings", "ModelDownloader", "System"]) as app:
    close_button = gr.Button("Close terminal")
    close_button.click(close_terminal, [], [], queue=False)

    folder_button = gr.Button("Finetuned-models")
    folder_button.click(open_finetuned_folder, [], [], queue=False)

    folder_button = gr.Button("Datasets")
    folder_button.click(open_datasets_folder, [], [], queue=False)

    folder_button = gr.Button("Outputs")
    folder_button.click(open_outputs_folder, [], [], queue=False)

    github_link = gr.HTML(
        '<div style="text-align: center; margin-top: 20px;">'
        '<a href="https://github.com/Dartvauder/NeuroTrainerWebUI" target="_blank" style="color: blue; text-decoration: none; font-size: 16px; margin-right: 20px;">'
        'GitHub'
        '</a>'
        '<a href="https://huggingface.co/Dartvauder007" target="_blank" style="color: blue; text-decoration: none; font-size: 16px;">'
        'Hugging Face'
        '</a>'
        '</div>'
    )

    app.launch(share=share_mode, server_name="localhost", auth=authenticate)
