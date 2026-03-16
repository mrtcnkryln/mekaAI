import json
import random
from pathlib import Path

RANDOM_SEED = 42
NEGATIVE_RATIO = 0.15

FILES = [
    Path("work/sft/train.jsonl"),
    Path("work/sft/dev.jsonl"),
    Path("work/sft/test.jsonl"),
]

NEGATIVE_QUESTIONS = [
    "Trafik cezasına nasıl itiraz edilir?",
    "İcra takibi nasıl başlatılır?",
    "Kira tespit davası hangi şartlarda açılır?",
    "İşten çıkarıldım, kıdem tazminatımı nasıl alır��m?",
    "Vergi borcu yapılandırması nasıl yapılır?",
    "Ceza davasında uzlaşma nedir?",
    "KVKK kapsamında açık rıza metni nasıl hazırlanır?",
    "Boşanma davasında nafaka nasıl belirlenir?",
    "Marka tescili nasıl yapılır?",
    "Şirket kuruluşu için gerekli adımlar nelerdir?"
]

SYSTEM = (
    "Sen yalnızca verilen dokümanlara dayanarak Türkçe yanıt veren bir mevzuat asistanısın. "
    "Cevaplarında mümkünse ilgili madde/bölüm dayanağını belirt. "
    "Dokümanlarda dayanak bulamazsan bunu açıkça belirt ve kesin hüküm verme."
)

NEGATIVE_ANSWER = (
    "Kısa cevap: Bu soruya yalnızca elimdeki dokümanlara dayanarak yanıt verebilirim; ilgili bir dayanak bulamadım.\n\n"
    "Dayanak: Bulunamadı (mevcut dokümanlar içinde ilgili madde/bölüm tespit edilemedi)\n\n"
    "Alıntı (kısa): Bulunamadı\n\n"
    "Açıklama: Soru, mevcut doküman setimin kapsamı dışında olabilir veya doğrudan dayanak içermiyor olabilir.\n"
    "Not: Genel bilgilendirme amaçlıdır; hukuki görüş değildir."
)

def load_jsonl(p: Path):
    data = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def write_jsonl(p: Path, data):
    with p.open("w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def make_negative_example(q: str):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": q},
            {"role": "assistant", "content": NEGATIVE_ANSWER},
        ]
    }

def main():
    random.seed(RANDOM_SEED)
    for p in FILES:
        data = load_jsonl(p)
        n = len(data)
        n_add = int(n * NEGATIVE_RATIO)

        negs = [make_negative_example(random.choice(NEGATIVE_QUESTIONS)) for _ in range(n_add)]
        data2 = data + negs
        random.shuffle(data2)

        write_jsonl(p, data2)
        print(f"[OK] {p}: {n} -> {len(data2)} (neg_added={n_add})")

if __name__ == "__main__":
    main()