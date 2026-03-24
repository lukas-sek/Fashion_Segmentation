import json
import shutil
from pathlib import Path
from typing import Set

VAL_JSON = Path("data/instances_attributes_val2020.json")
SOURCE_DIR = Path("data/test")
OUTPUT_DIR = Path("data/validation")


def load_val_stems(val_json_path: Path) -> Set[str]:
    with val_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {Path(img["file_name"]).stem for img in data.get("images", [])}

def main() -> None:
    val_json = VAL_JSON
    source_dir = SOURCE_DIR
    output_dir = OUTPUT_DIR

    if not val_json.exists():
        raise FileNotFoundError(f"Validation JSON not found: {val_json}")
    if not source_dir.exists() or not source_dir.is_dir():
        raise NotADirectoryError(f"Source folder not found or invalid: {source_dir}")

    val_stems = load_val_stems(val_json)
    source_files = [p for p in source_dir.iterdir() if p.is_file()]
    source_by_stem = {p.stem: p for p in source_files}

    missing = sorted(stem for stem in val_stems if stem not in source_by_stem)
    to_copy = [source_by_stem[stem] for stem in sorted(val_stems) if stem in source_by_stem]

    print(f"Validation images in JSON : {len(val_stems)}")
    print(f"Found in source folder    : {len(to_copy)}")
    print(f"Missing from source       : {len(missing)}")
    if missing:
        print("First 10 missing stems:")
        for stem in missing[:10]:
            print(f"  - {stem}")

    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src in to_copy:
        dst = output_dir / src.name
        shutil.copy2(src, dst)
        copied += 1

    print("\nDone.")
    print(f"Output folder : {output_dir}")
    print(f"Copied files  : {copied}")


if __name__ == "__main__":
    main()
