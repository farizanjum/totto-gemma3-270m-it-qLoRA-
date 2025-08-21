import json, random
from pathlib import Path
from datasets import load_dataset

MAX_ROWS, MAX_COLS = 40, 20

def linearize(ex):
    table = ex["table"]
    hl = set(tuple(x) for x in ex["highlighted_cells"])
    page = ex.get("table_page_title","") or ""
    sect = ex.get("table_section_title","") or ""
    meta = f"[PAGE] {page} [SECTION] {sect}".strip()
    lines = []
    for r, row in enumerate(table[:MAX_ROWS]):
        cells = []
        for c, cell in enumerate(row[:MAX_COLS]):
            val = (cell.get("value") or "").replace("\n"," ").strip()
            if (r,c) in hl: val = f"<hl>{val}</hl>"
            header = "H:" if cell.get("is_header") else ""
            cells.append(f"{header}C{c}={val if val else '<empty>'}")
        lines.append(" | ".join(cells))
    src = meta + "\n" + "\n".join(lines)

    target = ""
    for ann in ex.get("sentence_annotations", []):
        s = (ann.get("final_sentence") or "").strip()
        if s: target = s; break
    return {"text": src, "labels": target}

def main():
    ds = load_dataset("GEM/totto")
    proc = ds.map(linearize, remove_columns=ds["train"].column_names)
    proc = proc.filter(lambda x: len(x["text"])>0 and len(x["labels"])>0)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    proc["train"].to_json("data/processed/train.jsonl", lines=True, force_ascii=False)
    proc["validation"].to_json("data/processed/val.jsonl", lines=True, force_ascii=False)
    print("Saved: data/processed/train.jsonl, val.jsonl")
if __name__ == "__main__":
    main()
