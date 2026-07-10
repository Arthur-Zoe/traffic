from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import inspect_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect traffic sign dataset layout and labels")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--path-col", default="Path")
    parser.add_argument("--label-col", default="ClassId")
    parser.add_argument("--output", default="outputs/dataset_report.json")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv) if args.csv else None
    if csv_path is not None and not csv_path.is_absolute():
        csv_path = Path(args.data_dir) / csv_path
    report = inspect_dataset(
        data_dir=args.data_dir,
        csv_path=csv_path,
        path_col=args.path_col,
        label_col=args.label_col,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Images:", report["image_count"])
    print("Readable:", report["readable_images"])
    print("Corrupt:", report["corrupt_images"])
    print("Classes:", report["num_classes"])
    print("Recommended num_classes:", report["recommended_num_classes"])
    print("Saved:", output)


if __name__ == "__main__":
    main()
