import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = os.environ.get("BASE", "Qwen/Qwen2.5-7B-Instruct")
ADAPTER = os.environ.get("ADAPTER", "out/qwen25_7b_qlora_adapter")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "300"))

SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "Sen mevzuat temelli hukuki bilgi asistanısın. "
    "Cevaplarında mutlaka 'Dayanak:' ve 'Alıntı:' başlıkları olsun. "
    "Bilmiyorsan uydurma; 'Bilmiyorum' de."
)

def load():
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        dtype=torch.bfloat16,   # torch_dtype yerine dtype (uyarıyı keser)
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    return tok, model

tokenizer, model = load()

def build_prompt(history, user_text: str) -> str:
    # gr.ChatInterface history formatı: [(user, assistant), ...]
    p = f"SYSTEM: {SYSTEM_PROMPT}\n"
    for u, a in history:
        p += f"USER: {u}\nASSISTANT: {a}\n"
    p += f"USER: {user_text}\nASSISTANT:"
    return p

@torch.no_grad()
def respond(user_text, history):
    history = history or []
    prompt = build_prompt(history, user_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    answer = text.split("ASSISTANT:")[-1].strip()
    return answer

demo = gr.ChatInterface(
    fn=respond,
    title="mekaAI – Qwen2.5-7B + LoRA (RunPod Test UI)",
    description="Test amaçlıdır; hukuki görüş değildir.",
)

demo.launch(server_name="0.0.0.0", server_port=8888)