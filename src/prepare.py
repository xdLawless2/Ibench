import argparse
import json
from pathlib import Path
from typing import List, Dict


def collect_images(imgs_dir: Path) -> List[Path]:
    imgs = [p for p in imgs_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    imgs.sort(key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not imgs:
        raise SystemExit(f"No images found in {imgs_dir}")
    return imgs


def read_truth(truth_path: Path) -> List[int]:
    labels: List[int] = []
    with truth_path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            if "#" in t:
                t = t.split("#", 1)[0].strip()
                if not t:
                    continue
            labels.append(int(t))
    if not labels:
        raise SystemExit(f"No labels found in {truth_path}")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare IBench manifest from imgs and truth.")
    parser.add_argument("--imgs", type=Path, default=Path("imgs"))
    parser.add_argument("--truth", type=Path, default=Path("src/truth.txt"))
    parser.add_argument("--out", type=Path, default=Path("data/manifest.jsonl"))
    args = parser.parse_args()

    imgs = collect_images(args.imgs)
    truth = read_truth(args.truth)

    if len(imgs) != len(truth):
        raise SystemExit(f"Count mismatch: {len(imgs)} images vs {len(truth)} labels")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for idx, (img, label) in enumerate(zip(imgs, truth), start=1):
            rec: Dict = {"id": idx, "image_path": str(img), "label": label}
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote manifest: {args.out} ({len(truth)} items)")


if __name__ == "__main__":
    main()

