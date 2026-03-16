import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

IN_DIR = Path("work/clean_text")
OUT_PATH = Path("work/segments/segments.jsonl")

# MADDE başlıklarını yakalamak için:
# - MADDE 12 -
# - EK MADDE 1-
# - GEÇİCİ MADDE 2-
# - MADDE 6 - /A -   (biz bunu 6/A gibi normalize edeceğiz)
ARTICLE_HEADER_RE = re.compile(
    r"""(?mx)
    ^\s*
    (?P<kind>EK\s+MADDE|GEÇİCİ\s+MADDE|MADDE)
    \s+
    (?P<num>\d+)
    \s*
    (?P<suffix>(?:/\s*[A-Z])?)   # /A gibi
    \s*
    -\s*
    """,
    re.IGNORECASE,
)

def detect_kind(kind_raw: str) -> str:
    k = kind_raw.strip().upper()
    if k.startswith("EK"):
        return "ek_madde"
    if "GEÇ" in k:
        return "gecici_madde"
    return "madde"

def normalize_article_id(num: str, suffix: str, kind: str) -> str:
    sfx = (suffix or "").replace(" ", "")
    if sfx.startswith("/"):
        sfx = sfx[1:]
    base = f"{num}"
    if sfx:
        base = f"{num}/{sfx}"
    # ek/geçici maddelerde id'yi ayırt etmek için prefix ekliyoruz
    if kind == "ek_madde":
        return f"EK-{base}"
    if kind == "gecici_madde":
        return f"GECICI-{base}"
    return base

def split_articles(text: str) -> List[Tuple[Dict, str]]:
    """
    Dönen liste: [(header_meta, body_text), ...]
    header_meta: kind, num, suffix, start_index vs.
    """
    matches = list(ARTICLE_HEADER_RE.finditer(text))
    if not matches:
        return []

    parts = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        header = {
            "kind_raw": m.group("kind"),
            "num": m.group("num"),
            "suffix": m.group("suffix") or "",
            "header_text": text[m.start():m.end()].strip(),
            "span_start": start,
            "span_end": end,
        }
        body = text[m.end():end].strip()
        parts.append((header, body))
    return parts

def paragraph_segments(text: str, max_chars: int = 3000) -> List[str]:
    """
    Madde yapısı olmayan dokümanlar için paragraf bazlı bölme.
    max_chars: segment üst sınırı (çok uzun paragraf olursa böler)
    """
    # boş satırlarla paragrafları ayır
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    segs: List[str] = []
    buf = ""
    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= max_chars:
            buf = buf + "\n\n" + p
        else:
            segs.append(buf)
            buf = p
    if buf:
        segs.append(buf)
    return segs

def main():
    txts = sorted(IN_DIR.glob("*.txt"))
    if not txts:
        raise SystemExit("work/clean_text içinde .txt bulunamadı")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for p in txts:
            doc_id = p.stem
            text = p.read_text(encoding="utf-8", errors="ignore").strip()

            article_parts = split_articles(text)
            if article_parts:
                for header, body in article_parts:
                    kind = detect_kind(header["kind_raw"])
                    unit_id = normalize_article_id(header["num"], header["suffix"], kind)
                    rec = {
                        "doc_id": doc_id,
                        "unit_type": kind,              # madde / ek_madde / gecici_madde
                        "unit_id": unit_id,             # 12, 6/A, EK-1, GECICI-2 gibi
                        "header": header["header_text"],
                        "text": body,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total += 1
            else:
                # Genelge/rehber gibi: paragraf segmentlerine böl
                segs = paragraph_segments(text)
                for i, seg in enumerate(segs, start=1):
                    rec = {
                        "doc_id": doc_id,
                        "unit_type": "paragraf",
                        "unit_id": f"P-{i}",
                        "header": "",
                        "text": seg,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total += 1

    print(f"[OK] segments yazıldı: {OUT_PATH} (total_segments={total})")

if __name__ == "__main__":
    main()