import json
from pathlib import Path

FILES = [
    Path("work/sft/train.jsonl"),
    Path("work/sft/dev.jsonl"),
    Path("work/sft/test.jsonl"),
]

def main():
    for p in FILES:
        bad = 0
        total = 0
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    bad += 1
                    continue
                total += 1
                try:
                    obj = json.loads(line)
                except Exception:
                    print(f"[BAD JSON] {p} line={i}")
                    bad += 1
                    continue

                msgs = obj.get("messages", [])
                if len(msgs) != 3:
                    print(f"[BAD MSG COUNT] {p} line={i} count={len(msgs)}")
                    bad += 1
                    continue

                sys = msgs[0].get("content", "")
                user = msgs[1].get("content", "")
                assistant = msgs[2].get("content", "")

                if not user or not assistant or not sys:
                    print(f"[EMPTY FIELD] {p} line={i}")
                    bad += 1
                    continue

                # Negatif örnekler hariç çoğunda Dayanak ve Alıntı olmalı
                if "Dayanak:" not in assistant:
                    print(f"[NO REF] {p} line={i}")
                    bad += 1
                    continue

        print(f"[OK] {p}: total_lines={total} bad={bad}")

if __name__ == "__main__":
    main()