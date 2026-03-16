import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = os.environ.get("BASE", "Qwen/Qwen2.5-7B-Instruct")
ADAPTER = os.environ.get("ADAPTER", "out/qwen25_7b_qlora_adapter")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "250"))

SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "Sen mevzuat temelli hukuki bilgi asistanısın. "
    "Cevaplarında mutlaka 'Dayanak:' ve 'Alıntı:' başlıkları olsun. "
    "Bilmiyorsan uydurma; 'Bilmiyorum' de."
)

def load():
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    return tok, model

tokenizer, model = load()

def to_messages(history, user_text):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in (history or []):
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user_text})
    return msgs

@torch.no_grad()
def respond(user_text, history):
    messages = to_messages(history, user_text)

    # Qwen Instruct için doğru yöntem
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,          # loop’u kırmak için sample açıyoruz
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15, # en kritik
        no_repeat_ngram_size=4,  # ekstra emniyet
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)

    # Prompt kısmını kesip sadece yeni cevabı al
    answer = decoded[len(prompt):].strip()
    return answer

demo = gr.ChatInterface(fn=respond, title="mekaAI – Test UI")
demo.launch(server_name="0.0.0.0", server_port=8888)