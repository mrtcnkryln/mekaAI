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
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    return tok, model

tokenizer, model = load()

def build_prompt(history, user):
    p = f"SYSTEM: {SYSTEM_PROMPT}\n"
    for u, a in history:
        p += f"USER: {u}\nASSISTANT: {a}\n"
    p += f"USER: {user}\nASSISTANT:"
    return p

@torch.no_grad()
def respond(user, history):
    prompt = build_prompt(history, user)
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
    history = history + [(user, answer)]
    return "", history

with gr.Blocks() as demo:
    gr.Markdown("# mekaAI – Qwen2.5-7B + LoRA (RunPod Test UI)")
    gr.Markdown("Bu arayüz test amaçlıdır; hukuki görüş değildir.")
    chat = gr.Chatbot(height=520)
    msg = gr.Textbox(label="Soru", placeholder="Sorunu yaz ve Enter'a bas...")
    clear = gr.Button("Temizle")

    msg.submit(respond, [msg, chat], [msg, chat])
    clear.click(lambda: [], None, chat)

demo.launch(server_name="0.0.0.0", server_port=8888)