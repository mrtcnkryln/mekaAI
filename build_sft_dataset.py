import json
import random
import re
from pathlib import Path
from typing import Dict, List

SEGMENTS_PATH = Path("work/segments/segments.jsonl")
DOC_TITLES_PATH = Path("work/doc_titles.json")

OUT_TRAIN = Path("work/sft/train.jsonl")
OUT_DEV = Path("work/sft/dev.jsonl")
OUT_TEST = Path("work/sft/test.jsonl")

RANDOM_SEED = 42
DEV_RATIO = 0.1
TEST_RATIO = 0.1

SYSTEM = (
    "Sen yalnızca verilen dokümanlara dayanarak Türkçe yanıt veren bir mevzuat asistanısın. "
    "Cevaplarında mümkünse ilgili madde/bölüm dayanağını belirt. "
    "Dokümanlarda dayanak bulamazsan bunu açıkça belirt ve kesin hüküm verme."
)

DISCLAIMER = "Not: Genel bilgilendirme amaçlıdır; hukuki görüş değildir."

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def first_sentences(text: str, max_chars: int = 380) -> str:
    """
    Kısa alıntı için: metnin başından ~1-2 cümle veya max_chars kadar kırp.
    """
    t = normalize_spaces(text)
    if not t:
        return ""

    # 2. cümle sonuna kadar almaya çalış
    ends = [m.start() for m in re.finditer(r"[.!?]\s", t)]
    if len(ends) >= 2 and ends[1] > 60:
        snippet = t[: ends[1] + 1]
        return snippet.strip()

    # yoksa max_chars
    return t[:max_chars].strip()

def short_summary(text: str) -> str:
    # V1 özet: ilk cümle/iki cümle (sonra manuel altın set ile iyileştirilecek)
    return first_sentences(text, 280)

def load_doc_titles() -> Dict[str, str]:
    if not DOC_TITLES_PATH.exists():
        return {}
    return json.loads(DOC_TITLES_PATH.read_text(encoding="utf-8"))

def build_reference(seg: Dict, doc_titles: Dict[str, str]) -> str:
    doc_id = seg["doc_id"]
    doc_title = doc_titles.get(doc_id, doc_id)

    if seg["unit_type"] == "paragraf":
        return f"{doc_title} {seg['unit_id']}"
    if seg["unit_type"] == "madde":
        return f"{doc_title} Madde {seg['unit_id']}"
    if seg["unit_type"] == "ek_madde":
        return f"{doc_title} Ek Madde {seg['unit_id'].replace('EK-','')}"
    if seg["unit_type"] == "gecici_madde":
        return f"{doc_title} Geçici Madde {seg['unit_id'].replace('GECICI-','')}"
    return f"{doc_title} {seg['unit_type']} {seg['unit_id']}"

def make_answer(seg: Dict, doc_titles: Dict[str, str]) -> str:
    ref = build_reference(seg, doc_titles)
    text = seg["text"].strip()

    summary = short_summary(text)
    quote = first_sentences(text, 420)

    answer = (
        f"Kısa cevap: {summary}\n\n"
        f"Dayanak: {ref}\n\n"
        f"Alıntı (kısa): {quote}\n\n"
        f"Açıklama: Bu yanıt yalnızca verilen doküman metnine dayalı olarak üretilmiştir.\n"
        f"{DISCLAIMER}"
    )
    return answer

def make_qa_pairs(seg: Dict, doc_titles: Dict[str, str]) -> List[Dict]:
    ref_for_questions = build_reference(seg, doc_titles)
    doc_id = seg["doc_id"]
    text = seg["text"].strip()
    if not text:
        return []

    answer = make_answer(seg, doc_titles)

    qs = []
    if seg["unit_type"] == "paragraf":
        qs.append(f"{doc_id} belgesinde {seg['unit_id']} bölümünde ne anlatılıyor?")
        qs.append(f"{doc_id} kapsamında {seg['unit_id']} neyi açıklar?")
        qs.append(f"{doc_id} belgesine göre {seg['unit_id']} ile ilgili kural/ilke nedir?")
    else:
        qs.append(f"{ref_for_questions} neyi düzenler?")
        qs.append(f"{ref_for_questions} metninin özeti nedir?")
        # kullanıcı doc_id ile de sorabilir
        qs.append(f"{doc_id} içinde {seg['unit_id']} numaralı madde ne diyor?")

    out = []
    for q in qs:
        out.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": q},
                {"role": "assistant", "content": answer},
            ]
        })
    return out

def load_segments() -> List[Dict]:
    segs = []
    with SEGMENTS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            segs.append(json.loads(line))
    return segs

def write_jsonl(path: Path, items: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    random.seed(RANDOM_SEED)

    doc_titles = load_doc_titles()
    segs = load_segments()

    examples: List[Dict] = []
    for seg in segs:
        examples.extend(make_qa_pairs(seg, doc_titles))

    if not examples:
        raise SystemExit("Hiç eğitim örneği üretilemedi.")

    random.shuffle(examples)

    n = len(examples)
    n_test = int(n * TEST_RATIO)
    n_dev = int(n * DEV_RATIO)
    n_train = n - n_dev - n_test

    train = examples[:n_train]
    dev = examples[n_train:n_train+n_dev]
    test = examples[n_train+n_dev:]

    write_jsonl(OUT_TRAIN, train)
    write_jsonl(OUT_DEV, dev)
    write_jsonl(OUT_TEST, test)

    print(f"[OK] train={len(train)} dev={len(dev)} test={len(test)} toplam={n}")
    print(f"[OK] yazıldı: {OUT_TRAIN}, {OUT_DEV}, {OUT_TEST}")

if __name__ == "__main__":
    main()