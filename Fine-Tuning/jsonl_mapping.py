import json
import random
from pathlib import Path

DATASET_DIR = Path(r"C:\Users\gsart\Downloads\plantvillage\color")
BUCKET = "plant_disease_sdl"
OUTPUT_TRAIN = "plantvillage_train.jsonl"
OUTPUT_VAL = "plantvillage_val.jsonl"
SPLIT_RATIO = 0.93

def collect(dataset_root: Path):
    items = []
    for cls in dataset_root.iterdir():
        if cls.is_dir():
            label = cls.name
            for img in cls.rglob("*.*"):
                if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    items.append((img, label))
    return items

def jsonl_entry(img_path, label):
    gs_uri = f"gs://{BUCKET}/plantvillage/color/{img_path.parent.name}/{img_path.name}"
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "fileData": {
                            "mimeType": "image/jpeg",
                            "fileUri": gs_uri
                        }
                    },
                    {
                        "text": "What disease is shown in this image? Answer with only the class name."
                    }
                ]
            },
            {
                "role": "model",
                "parts": [
                    {"text": label}
                ]
            }
        ]
    }

def write_jsonl(data, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for img, label in data:
            f.write(json.dumps(jsonl_entry(img, label)) + "\n")

def main():
    all_items = collect(DATASET_DIR)
    random.shuffle(all_items)

    split = int(len(all_items) * SPLIT_RATIO)
    train, val = all_items[:split], all_items[split:]

    write_jsonl(train, OUTPUT_TRAIN)
    write_jsonl(val, OUTPUT_VAL)

if __name__ == "__main__":
    main()
