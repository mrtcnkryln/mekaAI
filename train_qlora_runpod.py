import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
TRAIN_FILE = os.environ.get("TRAIN_FILE", "work/export_hf/train.jsonl")
EVAL_FILE = os.environ.get("EVAL_FILE", "work/export_hf/dev.jsonl")
OUT_DIR = os.environ.get("OUT_DIR", "out/qwen25_7b_qlora_adapter")

MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "1024"))
EPOCHS = int(os.environ.get("EPOCHS", "1"))
LR = float(os.environ.get("LR", "2e-4"))

BATCH = int(os.environ.get("BATCH", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))

def main():
    assert torch.cuda.is_available(), "CUDA yok. L40S Pod çalışıyor mu kontrol et."

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    ds = load_dataset("json", data_files={"train": TRAIN_FILE, "eval": EVAL_FILE})

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Qwen/Llama family için iyi başlangıç:
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        report_to=[],
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=0.3,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        peft_config=lora,
        args=args,
    )

    trainer.train()
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"[OK] adapter kaydedildi: {OUT_DIR}")

if __name__ == "__main__":
    main()