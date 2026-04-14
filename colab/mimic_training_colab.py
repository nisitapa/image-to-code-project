# =========================================================
# Mimic Training Colab
# Fine-tuning a LLaVA-based Vision-Language Model with LoRA
# Task: UI image -> HTML with inline CSS
# =========================================================

# =========================
# INSTALL
# =========================
# Run these manually in Colab if needed:
# !pip install --quiet torch==2.9.0+cu126 torchvision==0.24.0+cu126 torchaudio==2.9.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
# !pip install --quiet fsspec==2025.3.0
# !pip install --quiet transformers datasets peft bitsandbytes accelerate scipy trl

# =========================
# IMPORTS
# =========================
import os
import gc
import json
import torch
from PIL import Image
from datasets import load_dataset
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)

# =========================
# MOUNT DRIVE
# =========================
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

# =========================
# CONFIG
# =========================
base_model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

# Change these paths to your actual files
dataset_path = "/content/drive/MyDrive/UI_Dataset/register_dataset.json"
output_dir = "/content/drive/MyDrive/UI_Dataset/outputs/model_register"
save_path = "/content/drive/MyDrive/UI_Dataset/Model/model_Regist"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(save_path, exist_ok=True)

# =========================
# QUANTIZATION CONFIG
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# =========================
# LOAD PROCESSOR + MODEL
# =========================
processor = LlavaNextProcessor.from_pretrained(base_model_id)

model = LlavaNextForConditionalGeneration.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)

# =========================
# LoRA CONFIG
# =========================
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================
# LOAD DATASET
# =========================
dataset = load_dataset("json", data_files=dataset_path)["train"]
dataset = dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print("Train size:", len(train_dataset))
print("Eval size:", len(eval_dataset))

# =========================
# PREPROCESS
# =========================
def preprocess_example(example):
    image = Image.open(example["image"]).convert("RGB")

    conversations = example["conversations"]
    prompt_text = conversations[0]["value"]

    processed = processor(
        text=prompt_text,
        images=image,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=1536,
    )

    processed = {k: v[0] for k, v in processed.items()}
    processed["labels"] = processed["input_ids"].clone()
    return processed

train_dataset = train_dataset.map(preprocess_example)
eval_dataset = eval_dataset.map(preprocess_example)

# =========================
# TRAINING ARGS
# =========================
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=True,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    report_to="none",
)

# =========================
# TRAINER
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# =========================
# TRAIN
# =========================
trainer.train()

# =========================
# SAVE ADAPTER
# =========================
trainer.model.save_pretrained(save_path)
processor.save_pretrained(save_path)

print("Saved to:", save_path)

# =========================
# CLEANUP
# =========================
gc.collect()
torch.cuda.empty_cache()
print("Training completed.")
