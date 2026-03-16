import json
from pathlib import Path

IN_FILES = {
    "train": Path("work/sft/train.jsonl"),
    "dev": Path("work/sft/dev.jsonl"),
    "test": Path("work/sft/test.jsonl"),
}

OUT_DIR = Path("work/export_hf")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    for split, path in IN_FILES.items():
        out_path = OUT_DIR / f"{split}.jsonl"
        n = 0
        with path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
            for line in fin:
                obj = json.loads(line)
                msgs = obj["messages"]
                sys = msgs[0]["content"].strip()
                user = msgs[1]["content"].strip()
                assistant = msgs[2]["content"].strip()

                text = f"SYSTEM: {sys}\nUSER: {user}\nASSISTANT: {assistant}\n"
                fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                n += 1
        print(f"[OK] {split}: {n} -> {out_path}")

if __name__ == "__main__":
    main()