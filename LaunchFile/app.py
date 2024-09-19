import logging
import os
import warnings
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)
cache_dir = os.path.join("cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = cache_dir
temp_dir = os.path.join("temp")
os.makedirs(temp_dir, exist_ok=True)
os.environ["TMPDIR"] = temp_dir
import markdown
import gc
import platform
import sys
import xformers
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


def print_system_info():
    print(f"NeuroSandboxWebUI")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {sys.platform}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"CPU cores: {os.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")
    print(f"Disk space: {psutil.disk_usage('/').total / (1024 ** 3):.2f} GB")

    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    else:
        print("CUDA available: No")

    print(f"PyTorch version: {torch.__version__}")
    print(f"Xformers version: {xformers.__version__}")


try:
    print_system_info()
except Exception as e:
    print(f"Unable to access system information: {e}")
    pass


def flush():
    gc.collect()
    torch.cuda.empty_cache()


def load_translation(lang):
    try:
        with open(f"translations/{lang}.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


translations = {
    "EN": {},
    "RU": load_translation("ru"),
    "ZH": load_translation("zh")
}


def _(text, lang="EN"):
    return translations[lang].get(text, text)


def load_settings():
    if not os.path.exists('Settings.json'):
        default_settings = {
            "language": "EN",
            "share_mode": False,
            "debug_mode": False,
            "monitoring_mode": False,
            "auto_launch": False,
            "show_api": False,
            "api_open": False,
            "queue_max_size": 10,
            "status_update_rate": "auto",
            "auth": {"username": "admin", "password": "admin"},
            "server_name": "localhost",
            "server_port": 7860,
            "hf_token": "",
            "theme": "Default",
            "custom_theme": {
                "enabled": False,
                "primary_hue": "red",
                "secondary_hue": "pink",
                "neutral_hue": "stone",
                "spacing_size": "spacing_md",
                "radius_size": "radius_md",
                "text_size": "text_md",
                "font": "Arial",
                "font_mono": "Courier New"
            }
        }
        with open('Settings.json', 'w') as f:
            json.dump(default_settings, f, indent=4)

    with open('Settings.json', 'r') as f:
        return json.load(f)


def save_settings(settings):
    with open('Settings.json', 'w') as f:
        json.dump(settings, f, indent=4)


def authenticate(username, password):
    settings = load_settings()
    auth = settings.get('auth', {})
    return username == auth.get('username') and password == auth.get('password')


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


def reload_model_lists():
    llm_models = get_available_llm_models()
    llm_lora_models = get_available_llm_lora_models()
    finetuned_llm_models = get_available_finetuned_llm_models()
    llm_datasets = get_available_llm_datasets()
    sd_models = get_available_sd_models()
    sd_lora_models = get_available_sd_lora_models()
    finetuned_sd_models = get_available_finetuned_sd_models()
    sd_datasets = get_available_sd_datasets()

    return [
        llm_models, llm_lora_models, finetuned_llm_models, llm_datasets,
        sd_models, sd_lora_models, finetuned_sd_models, sd_datasets
    ]


def reload_interface():
    updated_lists = reload_model_lists()[:8]
    return [gr.Dropdown(choices=list) for list in updated_lists]


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
                 weight_decay, warmup_steps, block_size, grad_accum_steps, adam_beta1, adam_beta2, adam_epsilon, lr_scheduler_type, freeze_layers, lora_r, lora_alpha, lora_dropout, use_xformers=False):
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

    if finetune_method == "Full" or "Freeze":
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
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        lr_scheduler_type=lr_scheduler_type
    )

    if use_xformers:
        try:
            import xformers.ops
            model.enable_xformers_memory_efficient_attention()
        except ImportError:
            print("xformers not installed. Proceeding without memory-efficient attention.")

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

    if finetune_method == "Freeze":
        num_freeze_layers = len(model.transformer.h) - freeze_layers
        for i, layer in enumerate(model.transformer.h):
            if i < num_freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

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
        predictions = [generate_text(model_name, lora_model_name, "transformers",
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
                learning_rate, lr_scheduler, lr_warmup_steps, max_train_steps, adam_beta1, adam_beta2, adam_weight_decay, adam_epsilon, max_grad_norm, noise_offset, rank, enable_xformers):
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
                f"--adam_beta1={adam_beta1}",
                f"--adam_beta2={adam_beta2}",
                f"--adam_weight_decay={adam_weight_decay}",
                f"--adam_epsilon={adam_epsilon}",
                f"--max_grad_norm={max_grad_norm}",
                f"--noise_offset={noise_offset}",
                f"--caption_column=text",
                f"--mixed_precision=no",
                f"--seed=0"
            ]
            if enable_xformers:
                args.append("--enable_xformers_memory_efficient_attention")
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
                f"--adam_beta1={adam_beta1}",
                f"--adam_beta2={adam_beta2}",
                f"--adam_weight_decay={adam_weight_decay}",
                f"--adam_epsilon={adam_epsilon}",
                f"--max_grad_norm={max_grad_norm}",
                f"--noise_offset={noise_offset}",
                f"--caption_column=text",
                f"--mixed_precision=no",
                f"--seed=0"
            ]
            if enable_xformers:
                args.append("--enable_xformers_memory_efficient_attention")
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
                f"--adam_beta1={adam_beta1}",
                f"--adam_beta2={adam_beta2}",
                f"--adam_weight_decay={adam_weight_decay}",
                f"--adam_epsilon={adam_epsilon}",
                f"--max_grad_norm={max_grad_norm}",
                f"--noise_offset={noise_offset}",
                f"--rank={rank}",
                f"--caption_column=text",
                f"--mixed_precision=no",
                f"--seed=0"
            ]
            if enable_xformers:
                args.append("--enable_xformers_memory_efficient_attention")
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
                f"--adam_beta1={adam_beta1}",
                f"--adam_beta2={adam_beta2}",
                f"--adam_weight_decay={adam_weight_decay}",
                f"--adam_epsilon={adam_epsilon}",
                f"--max_grad_norm={max_grad_norm}",
                f"--noise_offset={noise_offset}",
                f"--rank={rank}",
                f"--caption_column=text",
                f"--mixed_precision=no",
                f"--seed=0"
            ]
            if enable_xformers:
                args.append("--enable_xformers_memory_efficient_attention")
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

    disk = psutil.disk_usage('/')
    disk_total = f"{disk.total // (1024 ** 3)} GB"
    disk_free = f"{disk.free // (1024 ** 3)} GB"

    app_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    app_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                   for dirpath, dirnames, filenames in os.walk(app_folder)
                   for filename in filenames)
    app_size = f"{app_size // (1024 ** 3):.2f} GB"

    return (gpu_total_memory, gpu_used_memory, gpu_free_memory, gpu_temp, cpu_temp,
            ram_total, ram_used, ram_free, disk_total, disk_free, app_size)


def settings_interface(language, share_value, debug_value, monitoring_value, auto_launch, api_status, open_api, queue_max_size, status_update_rate, gradio_auth, server_name, server_port, hf_token, theme,
                       enable_custom_theme, primary_hue, secondary_hue, neutral_hue,
                       spacing_size, radius_size, text_size, font, font_mono):
    settings = load_settings()

    settings['language'] = language
    settings['share_mode'] = share_value == "True"
    settings['debug_mode'] = debug_value == "True"
    settings['monitoring_mode'] = monitoring_value == "True"
    settings['auto_launch'] = auto_launch == "True"
    settings['show_api'] = api_status == "True"
    settings['api_open'] = open_api == "True"
    settings['queue_max_size'] = int(queue_max_size) if queue_max_size else 10
    settings['status_update_rate'] = status_update_rate
    if gradio_auth:
        username, password = gradio_auth.split(':')
        settings['auth'] = {"username": username, "password": password}
    settings['server_name'] = server_name
    settings['server_port'] = int(server_port) if server_port else 7860
    settings['hf_token'] = hf_token
    settings['theme'] = theme
    settings['custom_theme']['enabled'] = enable_custom_theme
    settings['custom_theme']['primary_hue'] = primary_hue
    settings['custom_theme']['secondary_hue'] = secondary_hue
    settings['custom_theme']['neutral_hue'] = neutral_hue
    settings['custom_theme']['spacing_size'] = spacing_size
    settings['custom_theme']['radius_size'] = radius_size
    settings['custom_theme']['text_size'] = text_size
    settings['custom_theme']['font'] = font
    settings['custom_theme']['font_mono'] = font_mono

    save_settings(settings)

    message = "Settings updated successfully!"
    message += f"\nLanguage set to {settings['language']}"
    message += f"\nShare mode is {settings['share_mode']}"
    message += f"\nDebug mode is {settings['debug_mode']}"
    message += f"\nMonitoring mode is {settings['monitoring_mode']}"
    message += f"\nAutoLaunch mode is {settings['auto_launch']}"
    message += f"\nShow API mode is {settings['show_api']}"
    message += f"\nOpen API mode is {settings['api_open']}"
    message += f"\nQueue max size is {settings['queue_max_size']}"
    message += f"\nStatus update rate is {settings['status_update_rate']}"
    message += f"\nNew Gradio Auth is {settings['auth']}"
    message += f" Server will run on {settings['server_name']}:{settings['server_port']}"
    message += f"\nNew HF-Token is {settings['hf_token']}"
    message += f"\nTheme set to {theme and settings['custom_theme'] if enable_custom_theme else theme}"
    message += f"\nPlease restart the application for changes to take effect!"

    return message


def download_model(model_name_llm, model_name_sd):
    if not model_name_llm and not model_name_sd:
        return "Please select a model to download"

    if model_name_llm and model_name_sd:
        return "Please select one model type for downloading"

    if model_name_llm:
        model_url = ""
        if model_name_llm == "StarlingLM(Transformers7B)":
            model_url = "https://huggingface.co/Nexusflow/Starling-LM-7B-beta"
        elif model_name_llm == "OpenChat3.6(Llama8B.Q4)":
            model_url = "https://huggingface.co/bartowski/openchat-3.6-8b-20240522-GGUF/resolve/main/openchat-3.6-8b-20240522-Q4_K_M.gguf"
        model_path = os.path.join("inputs", "text", "llm_models", model_name_llm)

        if model_url:
            if model_name_llm == "StarlingLM(Transformers7B)":
                Repo.clone_from(model_url, model_path)
            else:
                response = requests.get(model_url, allow_redirects=True)
                with open(model_path, "wb") as file:
                    file.write(response.content)
            return f"LLM model {model_name_llm} downloaded successfully!"
        else:
            return "Invalid LLM model name"

    if model_name_sd:
        model_url = ""
        if model_name_sd == "Dreamshaper8(SD1.5)":
            model_url = "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors"
        elif model_name_sd == "RealisticVisionV4.0(SDXL)":
            model_url = "https://huggingface.co/SG161222/RealVisXL_V4.0/resolve/main/RealVisXL_V4.0.safetensors"
        model_path = os.path.join("inputs", "image", "sd_models", f"{model_name_sd}.safetensors")

        if model_url:
            response = requests.get(model_url, allow_redirects=True)
            with open(model_path, "wb") as file:
                file.write(response.content)
            return f"StableDiffusion model {model_name_sd} downloaded successfully!"
        else:
            return "Invalid StableDiffusion model name"


def get_wiki_content(url, local_file="Wikies/WikiEN.md"):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
    except:
        pass

    try:
        with open(local_file, 'r', encoding='utf-8') as file:
            content = file.read()
            return markdown.markdown(content)
    except:
        return "<p>Wiki content is not available.</p>"


settings = load_settings()
lang = settings['language']


llm_dataset_interface = gr.Interface(
    fn=create_llm_dataset,
    inputs=[
        gr.Dropdown(choices=get_available_llm_datasets(), label=_("Existing Dataset (optional)", lang)),
        gr.Textbox(label=_("Dataset Name", lang), type="text"),
        gr.Textbox(label=_("Instruction", lang), type="text"),
        gr.Textbox(label=_("Input", lang), type="text"),
        gr.Textbox(label=_("Output", lang), type="text"),
    ],
    outputs=[
        gr.Textbox(label=_("Status", lang), type="text"),
    ],
    title=_("NeuroTrainerWebUI (ALPHA) - LLM-Dataset", lang),
    description=_("Create a new dataset or add a new column to an existing dataset for LLM", lang),
    allow_flagging="never",
)

llm_finetune_interface = gr.Interface(
    fn=finetune_llm,
    inputs=[
        gr.Dropdown(choices=get_available_llm_models(), label=_("Model", lang)),
        gr.Dropdown(choices=get_available_llm_datasets(), label=_("Dataset", lang)),
        gr.Radio(choices=["Full", "Freeze", "LORA"], value="Full", label=_("Finetune Method", lang)),
        gr.Textbox(label=_("Output Model Name", lang), type="text"),
        gr.Number(value=10, label=_("Epochs", lang)),
        gr.Number(value=4, label=_("Batch size", lang)),
        gr.Number(value=3e-5, label=_("Learning rate", lang)),
        gr.Number(value=0.01, label=_("Weight decay", lang)),
        gr.Number(value=100, label=_("Warmup steps", lang)),
        gr.Number(value=128, label=_("Block size", lang)),
        gr.Number(value=1, label=_("Gradient accumulation steps", lang)),
        gr.Number(value=0.9, label=_("Adam beta 1", lang)),
        gr.Number(value=0.999, label=_("Adam beta 2", lang)),
        gr.Number(value=1e-8, label=_("Adam epsilon", lang)),
        gr.Dropdown(choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], value="linear", label=_("LR Scheduler", lang)),
        gr.Number(value=2, label=_("Freeze layers", lang)),
        gr.Number(value=16, label=_("LORA r", lang)),
        gr.Number(value=32, label=_("LORA alpha", lang)),
        gr.Number(value=0.05, label=_("LORA dropout", lang)),
        gr.Checkbox(label=_("Use xformers", lang), value=False),
    ],
    outputs=[
        gr.Textbox(label=_("Finetuning Status", lang), type="text"),
        gr.Plot(label=_("Finetuning Loss", lang))
    ],
    title=_("NeuroTrainerWebUI (ALPHA) - LLM-Finetune", lang),
    description=_("Finetune LLM models on a custom dataset", lang),
    allow_flagging="never",
)

llm_evaluate_interface = gr.Interface(
    fn=evaluate_llm,
    inputs=[
        gr.Dropdown(choices=get_available_finetuned_llm_models(), label=_("Model", lang)),
        gr.Dropdown(choices=get_available_llm_lora_models(), label=_("LORA Model (optional)", lang)),
        gr.Dropdown(choices=get_available_llm_datasets(), label=_("Dataset", lang)),
        gr.Textbox(label=_("Request", lang), type="text"),
        gr.Slider(minimum=1, maximum=2048, value=128, step=1, label=_("Max Length", lang)),
        gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label=_("Temperature", lang)),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, label=_("Top P", lang)),
        gr.Slider(minimum=0, maximum=100, value=20, step=1, label=_("Top K", lang)),
    ],
    outputs=[
        gr.Textbox(label=_("Evaluation Status", lang)),
        gr.Plot(label=_("Evaluation Metrics", lang)),
    ],
    title=_("NeuroTrainerWebUI (ALPHA) - LLM-Evaluate", lang),
    description=_("Evaluate finetuned LLM models on a custom dataset", lang),
    allow_flagging="never",
)

llm_quantize_interface = gr.Interface(
    fn=quantize_llm,
    inputs=[
        gr.Dropdown(choices=get_available_finetuned_llm_models(), label=_("Model", lang)),
        gr.Radio(choices=["Q2_K_M", "Q4_K_M", "Q6_K_M", "Q8_K_M"], value="Q4_K_M", label=_("Quantization Type", lang)),
    ],
    outputs=[
        gr.Textbox(label=_("Quantization Status", lang), type="text"),
    ],
    title=_("NeuroTrainerWebUI (ALPHA) - LLM-Quantize", lang),
    description=_("Quantize finetuned LLM models to .gguf format using llama.cpp", lang),
    allow_flagging="never",
)

llm_generate_interface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Dropdown(choices=get_available_finetuned_llm_models(), label=_("Model", lang)),
        gr.Dropdown(choices=get_available_llm_lora_models(), label=_("LORA Model (optional)", lang)),
        gr.Radio(choices=["transformers", "llama.cpp"], value="transformers", label=_("Model Type", lang)),
        gr.Textbox(label=_("Request", lang), type="text"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label=_("Max length", lang)),
        gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label=_("Temperature", lang)),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, label=_("Top P", lang)),
        gr.Slider(minimum=0, maximum=100, value=20, step=1, label=_("Top K", lang)),
        gr.Radio(choices=["txt", "json"], value="txt", label=_("Output Format", lang)),
    ],
    outputs=[
        gr.Textbox(label=_("Generated text", lang), type="text"),
        gr.Textbox(label=_("Message", lang), type="text"),
    ],
    title=_("NeuroTrainerWebUI (ALPHA) - LLM-Generate", lang),
    description=_("Generate text using finetuned LLM models", lang),
    allow_flagging="never",
)

sd_dataset_interface = gr.Interface(
    fn=create_sd_dataset,
    inputs=[
        gr.Files(label=_("Image Files", lang)),
        gr.Dropdown(choices=get_available_sd_datasets(), label=_("Existing Dataset (optional)", lang)),
        gr.Textbox(label=_("Dataset Name", lang)),
        gr.Textbox(label=_("Files Name", lang)),
        gr.Textbox(label=_("Prompt Text", lang))
    ],
    outputs=[
        gr.Textbox(label=_("Status", lang))
    ],
    title=_("NeuroTrainerWebUI (ALPHA) - StableDiffusion-Dataset", lang),
    description=_("Create a new dataset or add a new column to an existing dataset for Stable Diffusion", lang),
    allow_flagging="never",
)

sd_finetune_interface = gr.Interface(
    fn=finetune_sd,
    inputs=[
        gr.Dropdown(choices=get_available_sd_models(), label=_("Model", lang)),
        gr.Dropdown(choices=get_available_sd_datasets(), label=_("Dataset", lang)),
        gr.Radio(choices=["SD", "SDXL"], value="SD", label=_("Model Type", lang)),
        gr.Radio(choices=["Full", "LORA"], value="Full", label=_("Finetune Method", lang)),
        gr.Textbox(label=_("Output Model Name", lang), type="text"),
        gr.Number(value=512, label=_("Resolution", lang)),
        gr.Number(value=1, label=_("Train Batch Size", lang)),
        gr.Number(value=1, label=_("Gradient Accumulation Steps", lang)),
        gr.Number(value=5e-6, label=_("Learning Rate", lang)),
        gr.Dropdown(choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], value="linear", label=_("LR Scheduler", lang)),
        gr.Number(value=0, label=_("LR Warmup Steps", lang)),
        gr.Number(value=100, label=_("Max Train Steps", lang)),
        gr.Number(value=0.9, label=_("Adam beta 1", lang)),
        gr.Number(value=0.999, label=_("Adam beta 2", lang)),
        gr.Number(value=1e-2, label=_("Adam weight decay", lang)),
        gr.Number(value=1e-8, label=_("Adam epsilon", lang)),
        gr.Number(value=1.0, label=_("Max grad norm", lang)),
        gr.Number(value=0, label=_("Noise offset", lang)),
        gr.Number(value=4, label=_("LORA Rank", lang)),
        gr.Checkbox(label=_("Use xformers", lang), value=False),
    ],
    outputs=[
        gr.Textbox(label=_("Finetuning Status", lang), type="text"),
        gr.Plot(label=_("Finetuning Loss", lang))
    ],
    title=_("NeuroTrainerWebUI (ALPHA) - StableDiffusion-Finetune", lang),
    description=_("Finetune Stable Diffusion models on a custom dataset", lang),
    allow_flagging="never",
)

sd_evaluate_interface = gr.Interface(
    fn=evaluate_sd,
    inputs=[
        gr.Dropdown(choices=get_available_finetuned_sd_models(), label=_("Model", lang)),
        gr.Dropdown(choices=get_available_sd_lora_models(), label=_("LORA Model (optional)", lang)),
        gr.Dropdown(choices=get_available_sd_datasets(), label=_("Dataset", lang)),
        gr.Radio(choices=["Diffusers", "Safetensors"], value="Diffusers", label=_("Model Method", lang)),
        gr.Radio(choices=["SD", "SDXL"], value="SD", label=_("Model Type", lang)),
        gr.Textbox(label=_("Prompt", lang), type="text"),
        gr.Textbox(label=_("Negative Prompt", lang), type="text"),
        gr.Slider(minimum=1, maximum=150, value=30, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1, maximum=30, value=8, step=0.5, label=_("CFG", lang)),
    ],
    outputs=[
        gr.Textbox(label=_("Evaluation Status", lang)),
        gr.Plot(label=_("Evaluation Metrics", lang)),
    ],
    title=_("NeuroTrainerWebUI (ALPHA) - StabledDiffusion-Evaluate", lang),
    description=_("Evaluate finetuned Stable Diffusion models on a custom dataset", lang),
    allow_flagging="never",
)

sd_convert_interface = gr.Interface(
    fn=convert_sd_model_to_safetensors,
    inputs=[
        gr.Dropdown(choices=get_available_finetuned_sd_models(), label=_("Model", lang)),
        gr.Radio(choices=["SD", "SDXL"], value="SD", label=_("Model Type", lang)),
        gr.Checkbox(label=_("Use Half Precision", lang), value=False),
        gr.Checkbox(label=_("Use Safetensors", lang), value=False),
    ],
    outputs=[
        gr.Textbox(label=_("Conversion Status", lang), type="text"),
    ],
    title=_("NeuroTrainerWebUI (ALPHA) - StableDiffusion-Conversion", lang),
    description=_("Convert finetuned Stable Diffusion models to single file (.ckpt or .safetensors)", lang),
    allow_flagging="never",
)

sd_generate_interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Dropdown(choices=get_available_finetuned_sd_models(), label=_("Model", lang)),
        gr.Dropdown(choices=get_available_sd_lora_models(), label=_("LORA Model (optional)", lang)),
        gr.Radio(choices=["Diffusers", "Safetensors"], value="Diffusers", label=_("Model Method", lang)),
        gr.Radio(choices=["SD", "SDXL"], value="SD", label=_("Model Type", lang)),
        gr.Textbox(label=_("Prompt", lang), type="text"),
        gr.Textbox(label=_("Negative Prompt", lang), type="text"),
        gr.Slider(minimum=1, maximum=150, value=30, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1, maximum=30, value=8, step=0.5, label=_("CFG", lang)),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label=_("Width", lang)),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label=_("Height", lang)),
        gr.Radio(choices=["png", "jpeg"], value="png", label=_("Output Format", lang)),
    ],
    outputs=[
        gr.Image(label=_("Generated Image", lang)),
        gr.Textbox(label=_("Message", lang), type="text"),
    ],
    title=_("NeuroTrainerWebUI (ALPHA) - StableDiffusion-Generate", lang),
    description=_("Generate images using finetuned Stable Diffusion models", lang),
    allow_flagging="never",
)

wiki_interface = gr.Interface(
    fn=get_wiki_content,
    inputs=[
        gr.Textbox(label=_("Online Wiki", lang), value=(
            "https://github.com/Dartvauder/NeuroTrainerWebUI/wiki/EN‐Wiki" if lang == "EN" else
            "https://github.com/Dartvauder/NeuroTrainerWebUI/wiki/ZH‐Wiki" if lang == "ZH" else
            "https://github.com/Dartvauder/NeuroTrainerWebUI/wiki/RU‐Wiki"
        ), interactive=False),
        gr.Textbox(label=_("Local Wiki", lang), value=(
            "Wikies/WikiEN.md" if lang == "EN" else
            "Wikies/WikiZH.md" if lang == "ZH" else
            "Wikies/WikiRU.md"
        ), interactive=False)
    ],
    outputs=gr.HTML(label=_("Wiki Content", lang)),
    title=_("NeuroTrainerWebUI (ALPHA) - Wiki", lang),
    description=_("This interface displays the Wiki content from the specified URL or local file.", lang),
    allow_flagging="never",
    clear_btn=None,
)

model_downloader_interface = gr.Interface(
    fn=download_model,
    inputs=[
        gr.Dropdown(choices=[None, "StarlingLM(Transformers7B)", "OpenChat3.6(Llama8B.Q4)"], label=_("Download LLM model", lang), value=None),
        gr.Dropdown(choices=[None, "Dreamshaper8(SD1.5)", "RealisticVisionV4.0(SDXL)"], label=_("Download StableDiffusion model", lang), value=None),
    ],
    outputs=[
        gr.Textbox(label=_("Message", lang), type="text"),
    ],
    title=_("NeuroTrainerWebUI (ALPHA) - ModelDownloader", lang),
    description=_("This user interface allows you to download LLM and StableDiffusion models", lang),
    allow_flagging="never",
)

settings_interface = gr.Interface(
    fn=settings_interface,
    inputs=[
        gr.Radio(choices=["EN", "RU", "ZH"], label=_("Language", lang), value=settings['language']),
        gr.Radio(choices=["True", "False"], label=_("Share Mode", lang), value="False"),
        gr.Radio(choices=["True", "False"], label=_("Debug Mode", lang), value="False"),
        gr.Radio(choices=["True", "False"], label=_("Monitoring Mode", lang), value="False"),
        gr.Radio(choices=["True", "False"], label=_("Enable AutoLaunch", lang), value="False"),
        gr.Radio(choices=["True", "False"], label=_("Show API", lang), value="False"),
        gr.Radio(choices=["True", "False"], label=_("Open API", lang), value="False"),
        gr.Number(label=_("Queue max size", lang), value=settings['queue_max_size']),
        gr.Textbox(label=_("Queue status update rate", lang), value=settings['status_update_rate']),
        gr.Textbox(label=_("Gradio Auth", lang), value=settings['auth']),
        gr.Textbox(label=_("Server Name", lang), value=settings['server_name']),
        gr.Number(label=_("Server Port", lang), value=settings['server_port']),
        gr.Textbox(label=_("Hugging Face Token", lang), value=settings['hf_token'])
    ],
    additional_inputs=[
        gr.Radio(choices=["Base", "Default", "Glass", "Monochrome", "Soft"], label=_("Theme", lang), value=settings['theme']),
        gr.Checkbox(label=_("Enable Custom Theme", lang), value=settings['custom_theme']['enabled']),
        gr.Textbox(label=_("Primary Hue", lang), value=settings['custom_theme']['primary_hue']),
        gr.Textbox(label=_("Secondary Hue", lang), value=settings['custom_theme']['secondary_hue']),
        gr.Textbox(label=_("Neutral Hue", lang), value=settings['custom_theme']['neutral_hue']),
        gr.Radio(choices=["spacing_sm", "spacing_md", "spacing_lg"], label=_("Spacing Size", lang), value=settings['custom_theme'].get('spacing_size', 'spacing_md')),
        gr.Radio(choices=["radius_none", "radius_sm", "radius_md", "radius_lg"], label=_("Radius Size", lang), value=settings['custom_theme'].get('radius_size', 'radius_md')),
        gr.Radio(choices=["text_sm", "text_md", "text_lg"], label=_("Text Size", lang), value=settings['custom_theme'].get('text_size', 'text_md')),
        gr.Textbox(label=_("Font", lang), value=settings['custom_theme'].get('font', 'Arial')),
        gr.Textbox(label=_("Monospaced Font", lang), value=settings['custom_theme'].get('font_mono', 'Courier New'))
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Theme builder", lang), open=False),
    outputs=[
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroTrainerWebUI (ALPHA) - Settings", lang),
    description=_("This user interface allows you to change settings of the application", lang),
    allow_flagging="never",
)

system_interface = gr.Interface(
    fn=get_system_info,
    inputs=[],
    outputs=[
        gr.Textbox(label=_("GPU Total Memory", lang)),
        gr.Textbox(label=_("GPU Used Memory", lang)),
        gr.Textbox(label=_("GPU Free Memory", lang)),
        gr.Textbox(label=_("GPU Temperature", lang)),
        gr.Textbox(label=_("CPU Temperature", lang)),
        gr.Textbox(label=_("RAM Total", lang)),
        gr.Textbox(label=_("RAM Used", lang)),
        gr.Textbox(label=_("RAM Free", lang)),
        gr.Textbox(label=_("Disk Total Space", lang)),
        gr.Textbox(label=_("Disk Free Space", lang)),
        gr.Textbox(label=_("Application Folder Size", lang)),
    ],
    title=_("NeuroTrainerWebUI (ALPHA) - System", lang),
    description=_("This interface displays system information", lang),
    allow_flagging="never",
)

if settings['custom_theme']['enabled']:
    theme = getattr(gr.themes, settings['theme'])(
        primary_hue=settings['custom_theme']['primary_hue'],
        secondary_hue=settings['custom_theme']['secondary_hue'],
        neutral_hue=settings['custom_theme']['neutral_hue'],
        spacing_size=getattr(gr.themes.sizes, settings['custom_theme']['spacing_size']),
        radius_size=getattr(gr.themes.sizes, settings['custom_theme']['radius_size']),
        text_size=getattr(gr.themes.sizes, settings['custom_theme']['text_size']),
        font=settings['custom_theme']['font'],
        font_mono=settings['custom_theme']['font_mono']
    )
else:
    theme = getattr(gr.themes, settings['theme'])()

with gr.TabbedInterface([
    gr.TabbedInterface([llm_dataset_interface, llm_finetune_interface, llm_evaluate_interface, llm_quantize_interface, llm_generate_interface],
                       tab_names=[_("Dataset", lang), _("Finetune", lang), _("Evaluate", lang), _("Quantize", lang), _("Generate", lang)]),
    gr.TabbedInterface([sd_dataset_interface, sd_finetune_interface, sd_evaluate_interface, sd_convert_interface, sd_generate_interface],
                       tab_names=[_("Dataset", lang), _("Finetune", lang), _("Evaluate", lang), _("Conversion", lang), _("Generate", lang)]),
    gr.TabbedInterface([wiki_interface, model_downloader_interface, settings_interface, system_interface],
                       tab_names=[_("Wiki", lang), _("ModelDownloader", lang), _("Settings", lang), _("System", lang)])
],
    tab_names=[_("LLM", lang), _("StableDiffusion", lang), _("Interface", lang)], theme=theme) as app:
    reload_button = gr.Button(_("Reload interface", lang))

    close_button = gr.Button(_("Close terminal", lang))
    close_button.click(close_terminal, [], [], queue=False)

    folder_button = gr.Button(_("Finetuned-models", lang))
    folder_button.click(open_finetuned_folder, [], [], queue=False)

    folder_button = gr.Button(_("Datasets", lang))
    folder_button.click(open_datasets_folder, [], [], queue=False)

    folder_button = gr.Button(_("Outputs", lang))
    folder_button.click(open_outputs_folder, [], [], queue=False)

    dropdowns_to_update = [
        llm_dataset_interface.input_components[0],
        llm_finetune_interface.input_components[0],
        llm_finetune_interface.input_components[1],
        llm_evaluate_interface.input_components[0],
        llm_evaluate_interface.input_components[1],
        llm_evaluate_interface.input_components[2],
        llm_quantize_interface.input_components[0],
        llm_generate_interface.input_components[0],
        llm_generate_interface.input_components[1],
        sd_dataset_interface.input_components[1],
        sd_finetune_interface.input_components[0],
        sd_finetune_interface.input_components[1],
        sd_evaluate_interface.input_components[0],
        sd_evaluate_interface.input_components[1],
        sd_evaluate_interface.input_components[2],
        sd_convert_interface.input_components[0],
        sd_generate_interface.input_components[0],
        sd_generate_interface.input_components[1],
    ]

    reload_button.click(reload_interface, outputs=dropdowns_to_update[:8])

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

    app.queue(api_open=settings['api_open'], max_size=settings['queue_max_size'],
              status_update_rate=settings['status_update_rate'])
    app.launch(
        share=settings['share_mode'],
        debug=settings['debug_mode'],
        enable_monitoring=settings['monitoring_mode'],
        inbrowser=settings['auto_launch'],
        show_api=settings['show_api'],
        auth=authenticate if settings['auth'] else None,
        server_name=settings['server_name'],
        server_port=settings['server_port'],
        favicon_path="project-image.png",
        auth_message=_("Welcome to NeuroTrainerWebUI! (ALPHA)", lang)
    )