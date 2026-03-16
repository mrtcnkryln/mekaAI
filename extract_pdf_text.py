import os
from pathlib import Path
import fitz  # PyMuPDF

DATASETS_DIR = Path("datasets")
OUT_DIR = Path("work/raw_text")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    parts = []
    for page in doc:
        t = page.get_text("text")
        if t:
            parts.append(t)
    return "\n".join(parts)

def main():
    pdfs = sorted(DATASETS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("datasets klasöründe .pdf bulunamadı.")

    for pdf in pdfs:
        txt = extract_text(pdf)
        out_path = OUT_DIR / (pdf.stem + ".txt")
        out_path.write_text(txt, encoding="utf-8")
        print(f"[OK] {pdf.name} -> {out_path}")

if __name__ == "__main__":
    main()