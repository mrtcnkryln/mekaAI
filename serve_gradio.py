import os, json
from pathlib import Path
import numpy as np
import faiss
import torch
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -------- Config
BASE = os.environ.get("BASE", "Qwen/Qwen2.5-7B-Instruct")
ADAPTER = os.environ.get("ADAPTER", "out/qwen25_7b_qlora_adapter")
USE_ADAPTER = os.environ.get("USE_ADAPTER", "1") == "1"

INDEX_DIR = Path(os.environ.get("INDEX_DIR", "index"))
EMB_MODEL = os.environ.get("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
TOP_K = int(os.environ.get("TOP_K", "4"))

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "220"))

SYSTEM_PROMPT = (
    "Sen Türkiye mevzuatına dayalı bir asistansın.\n"
    "Aşağıdaki ALINTI parçaları tek kaynaktır.\n"
    "Kurallar:\n"
    "- Dayanak ve Alıntı sadece ALINTI'dan gelmeli.\n"
    "- ALINTI içinde yoksa: 'Bilmiyorum (kaynakta yok)'.\n"
    "- Yorum bölümünde pratik açıklama yapabilirsin ama yeni madde numarası UYDURMA.\n\n"
    "Çıktı formatı:\n"
    "Kısa cevap: ...\n"
    "Dayanak: ...\n"
    "Alıntı: ...\n"
    "Yorum: ...\n"
)

# -------- Load retrieval index
embedder = SentenceTransformer(EMB_MODEL)
index = faiss.read_index(str(INDEX_DIR / "laws.faiss"))
meta = json.loads((INDEX_DIR / "laws_meta.json").read_text(encoding="utf-8"))

def retrieve(query: str, k: int = TOP_K):
    q = embedder.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")
    scores, ids = index.search(q, k)
    hits = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx == -1:
            continue
        hits.append((float(score), meta[idx]))
    return hits

# -------- Load LLM
tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    dtype=torch.bfloat16,
    device_map="auto",
)
if USE_ADAPTER:
    model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

def make_prompt(user_text: str, history):
    hits = retrieve(user_text, TOP_K)

    alinti_blocks = []
    for score, m in hits:
        header = f"[{m['file']} | madde={m.get('article_no')} | score={score:.3f}]"
        alinti_blocks.append(header + "\n" + m["text"])
    alinti = "\n\n---\n\n".join(alinti_blocks)

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in (history or []):
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": f"SORU: {user_text}\n\nALINTI:\n{alinti}\n"})

    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def respond(user_text, history):
    prompt = make_prompt(user_text, history)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=1.12,
        no_repeat_ngram_size=4,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()

demo = gr.ChatInterface(
    fn=respond,
    title="mekaAI – RAG (work/clean_text)",
    description="Dayanak/Alıntı yalnızca work/clean_text içindeki metinlerden gelir.",
)
demo.launch(server_name="0.0.0.0", server_port=8888)