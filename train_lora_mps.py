import os
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ---- AYARLAR ----
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
TRAIN_FILE = "work/export_hf/train.jsonl"
EVAL_FILE = "work/export_hf/dev.jsonl"
OUT_DIR = "out/lora_adapter"

# Mac için makul başlangıç değerleri
MAX_SEQ_LEN = 768
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 2e-4
EPOCHS = 1

def main():
    assert torch.backends.mps.is_available(), "MPS yok görünüyor. PyTorch kurulumunu kontrol et."

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # MPS'de fp16/bf16 bazen sorun çıkarabiliyor; güvenli başlangıç fp32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map={"": "mps"},
    )

    # LoRA config (başlangıç)
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Qwen/Llama için genelde bu target modüller iş görür; gerekirse düzeltiriz
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files={"train": TRAIN_FILE, "eval": EVAL_FILE})

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        report_to=[],
        fp16=False,
        bf16=False,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=args,
        data_collator=collator,
        packing=False,
    )

    trainer.train()
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"[OK] adapter kaydedildi: {OUT_DIR}")

if __name__ == "__main__":
    main()