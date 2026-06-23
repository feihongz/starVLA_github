import argparse
import json
import time
from pathlib import Path


MODES = ("expert", "fast", "decoded")


def iter_json_objects(text: str):
    decoder = json.JSONDecoder()
    pos = 0
    while True:
        start = text.find("{", pos)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            pos = start + 1
            continue
        yield obj
        pos = start + end


def parse_records(log_path: Path):
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    records = []
    for record in iter_json_objects(text):
        if isinstance(record, dict) and {"task", "episode", "mode", "success"} <= set(record):
            records.append(record)
    return records


def build_progress(root: Path, expected_per_mode: int):
    logs = sorted((root / "logs").glob("chunk*_gpu*.log"))
    summary = {mode: {"done": 0, "succ": 0} for mode in MODES}
    unique_records = {}
    chunks = []

    for log_path in logs:
        records = parse_records(log_path)
        for record in records:
            if record.get("counted", True) is False:
                continue
            if not ({"task", "episode", "mode", "success"} <= set(record)):
                continue
            key = (str(record["task"]), int(record["episode"]), str(record["mode"]))
            unique_records[key] = record

        counted_records = [
            record
            for record in records
            if record.get("counted", True) is not False and {"task", "episode", "mode", "success"} <= set(record)
        ]
        chunks.append((log_path.name, counted_records))

    for record in unique_records.values():
            mode = record["mode"]
            if mode not in summary:
                summary[mode] = {"done": 0, "succ": 0}
            summary[mode]["done"] += 1
            summary[mode]["succ"] += int(bool(record["success"]))

    lines = [
        "updated_at\t" + time.strftime("%F %T"),
        "mode\tdone\tsuccess\trate\tremain",
    ]
    for mode in MODES:
        item = summary.get(mode, {"done": 0, "succ": 0})
        rate = item["succ"] / item["done"] if item["done"] else 0.0
        lines.append(
            f"{mode}\t{item['done']}\t{item['succ']}\t{rate:.3f}\t"
            f"{expected_per_mode - item['done']}"
        )

    lines.extend(["", "chunk\tdone\texpert\tfast\tdecoded\tlast"])
    for name, records in chunks:
        counts = {mode: sum(1 for record in records if record["mode"] == mode) for mode in MODES}
        last = records[-1] if records else None
        last_text = "none"
        if last:
            last_text = (
                f"{last.get('task')} ep{last.get('episode')} {last.get('mode')} "
                f"success={last.get('success')} steps={last.get('steps')}"
            )
        lines.append(
            f"{name}\t{len(records)}\t{counts['expert']}\t{counts['fast']}\t"
            f"{counts['decoded']}\t{last_text}"
        )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--expected_per_mode", type=int, default=250)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or args.root / "progress.tsv"
    text = build_progress(args.root, args.expected_per_mode)
    output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
