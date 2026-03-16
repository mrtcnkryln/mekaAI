import re
from pathlib import Path

IN_DIR = Path("work/raw_text")
OUT_DIR = Path("work/clean_text")

# Sık görülen gürültü satırlarını yakalamak için bazı kalıplar.
# (Bu listeyi senin dosyalara göre sonra genişletebiliriz.)
NOISE_LINE_PATTERNS = [
    r"^Sayfa\s+\d+\s*/\s*\d+\s*$",
    r"^\s*\d+\s*$",
    r"^T\.C\.\s*$",
    r"^Resm[iî]\s+Gazete\s*$",
    r"^www\.mevzuat\.gov\.tr\s*$",
    r"^mevzuat\.gov\.tr\s*$",
]

NOISE_LINE_RE = re.compile("|".join(f"(?:{p})" for p in NOISE_LINE_PATTERNS), re.IGNORECASE)

def normalize_dashes(s: str) -> str:
    # farklı tire/uzun tire karakterlerini standardize et
    return s.replace("–", "-").replace("—", "-").replace("−", "-")

def normalize_whitespace(s: str) -> str:
    # çoklu boşlukları azalt, satır sonlarını normalize et
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # bazı PDF’lerde " \n" gibi boşluklar olur
    s = re.sub(r"[ \t]+\n", "\n", s)
    # 3+ boş satırı 2 boş satıra indir
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s

def normalize_madde_headers(s: str) -> str:
    """
    MADDE başlıklarını olabildiğince tek forma yaklaştır:
    - "Madde 1 –" / "MADDE 1-" / "MADDE 1 -" -> "MADDE 1 -"
    - "MADDE 1." -> "MADDE 1 -"
    """
    # Önce tireleri standardize edelim
    s = normalize_dashes(s)

    # "Madde" kelimesini başta yakala (satır başı)
    # Örnek yakalamalar:
    #   Madde 1 –
    #   MADDE 12-
    #   MADDE 3 .
    #   MADDE 4 –
    madde_re = re.compile(
        r"(?m)^\s*(MADDE|Madde)\s+(\d+[A-Za-z]?)\s*([.\-:]|\s-\s|-\s|–\s|—\s|)\s*"
    )
    s = madde_re.sub(lambda m: f"MADDE {m.group(2)} - ", s)
    return s

def drop_noise_lines(s: str) -> str:
    lines = s.split("\n")
    kept = []
    for line in lines:
        ln = line.strip()
        if not ln:
            kept.append("")
            continue
        if NOISE_LINE_RE.match(ln):
            continue
        kept.append(line)
    return "\n".join(kept)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    txts = sorted(IN_DIR.glob("*.txt"))
    if not txts:
        raise SystemExit("work/raw_text içinde .txt bulunamadı")

    for p in txts:
        raw = p.read_text(encoding="utf-8", errors="ignore")
        s = raw
        s = normalize_whitespace(s)
        s = drop_noise_lines(s)
        s = normalize_whitespace(s)
        s = normalize_madde_headers(s)
        s = normalize_whitespace(s)

        out = OUT_DIR / p.name
        out.write_text(s, encoding="utf-8")
        print(f"[OK] {p.name} -> {out.name} (chars={len(s)})")

if __name__ == "__main__":
    main()