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

def build_prompt(messages, user_text: str):
    # messages: [{"role": "user"/"assistant", "content": "..."}]
    p = f"SYSTEM: {SYSTEM_PROMPT}\n"
    for m in messages:
        if m["role"] == "user":
            p += f"USER: {m['content']}\n"
        elif m["role"] == "assistant":
            p += f"ASSISTANT: {m['content']}\n"
    p += f"USER: {user_text}\nASSISTANT:"
    return p

@torch.no_grad()
def respond(user_text, messages):
    if messages is None:
        messages = []

    prompt = build_prompt(messages, user_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    answer = text.split("ASSISTANT:")[-1].strip()

    messages = messages + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": answer},
    ]
    return "", messages

with gr.Blocks() as demo:
    gr.Markdown("# mekaAI – Qwen2.5-7B + LoRA (RunPod Test UI)")
    gr.Markdown("Bu arayüz test amaçlıdır; hukuki görüş değildir.")

    chatbot = gr.Chatbot(type="messages", height=520)
    msg = gr.Textbox(label="Soru", placeholder="Sorunu yaz ve Enter'a bas...")
    clear = gr.Button("Temizle")

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: [], None, chatbot)

demo.launch(server_name="0.0.0.0", server_port=8888)