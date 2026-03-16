import os, re, json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

TEXT_DIR = Path(os.environ.get("TEXT_DIR", "work/clean_text"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "index"))
EMB_MODEL = os.environ.get("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize(t: str) -> str:
    t = t.replace("\r\n", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def split_madde(text: str):
    """
    Metni mümkünse 'MADDE 15' benzeri başlıklara göre böler.
    Bulamazsa sabit chunk + overlap yapar.
    """
    text = normalize(text)
    lines = text.splitlines()

    # Yaklaşık madde başlıkları: "MADDE 15", "Madde 15", "MADDE 15-"
    pat = re.compile(r"(?im)^\s*(madde|md\.?|m\.)\s*(\d+)\s*([\-–].*)?$")

    cuts = []
    for i, line in enumerate(lines):
        if pat.match(line.strip()):
            cuts.append(i)

    chunks = []
    if len(cuts) >= 3:
        cuts.append(len(lines))
        for a, b in zip(cuts, cuts[1:]):
            block = "\n".join(lines[a:b]).strip()
            m = pat.match(lines[a].strip())
            art = m.group(2) if m else None
            if len(block) >= 250:
                chunks.append(("madde", art, block))
        return chunks

    # Fallback: sabit chunk + overlap
    step = 1200
    overlap = 200
    i = 0
    while i < len(text):
        block = text[i:i+step]
        if len(block.strip()) >= 250:
            chunks.append(("chunk", None, block.strip()))
        i += (step - overlap)
    return chunks

def main():
    files = sorted(TEXT_DIR.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found under {TEXT_DIR.resolve()}")

    embedder = SentenceTransformer(EMB_MODEL)

    items = []
    for p in files:
        content = p.read_text(encoding="utf-8", errors="ignore")
        for kind, art, chunk in split_madde(content):
            items.append({
                "file": p.name,
                "kind": kind,
                "article_no": art,
                "text": chunk
            })

    texts = [it["text"] for it in items]
    embs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embs = np.asarray(embs, dtype="float32")

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine sim için normalize + inner product
    index.add(embs)

    faiss.write_index(index, str(OUT_DIR / "laws.faiss"))
    (OUT_DIR / "laws_meta.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK: indexed_chunks={len(items)} dim={dim} from_files={len(files)} out={OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()