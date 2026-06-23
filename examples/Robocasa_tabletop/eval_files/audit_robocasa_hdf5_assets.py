"""Audit RoboCasa HDF5 model XML asset references against the local asset tree."""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py


def _rewrite_asset_path(path: str, assets_root: Path) -> Path | None:
    match = re.search(r"models/assets/(.+)$", path)
    if not match:
        return None
    return assets_root / match.group(1)


def _collect_xml_file_refs(xml_text: str) -> list[str]:
    root = ET.fromstring(xml_text)
    refs: list[str] = []
    for elem in root.iter():
        for attr in ("file", "texture", "mesh"):
            value = elem.attrib.get(attr)
            if value and ("/" in value or "\\" in value):
                refs.append(value)
    return refs


def audit_hdf5_file(path: Path, assets_root: Path, max_demos: int | None) -> dict[str, Any]:
    checked = 0
    missing_by_ref: Counter[str] = Counter()
    missing_by_demo: dict[str, list[str]] = {}
    refs_by_demo: dict[str, int] = {}
    parse_errors: dict[str, str] = {}

    with h5py.File(path, "r") as handle:
        demo_keys = sorted(handle["data"].keys(), key=lambda name: int(name.split("_")[-1]))
        if max_demos is not None:
            demo_keys = demo_keys[:max_demos]
        for demo_key in demo_keys:
            checked += 1
            xml_text = handle["data"][demo_key].attrs.get("model_file", "")
            try:
                refs = _collect_xml_file_refs(xml_text)
            except Exception as exc:  # noqa: BLE001
                parse_errors[demo_key] = repr(exc)
                continue
            refs_by_demo[demo_key] = len(refs)
            missing: list[str] = []
            for ref in refs:
                local_path = _rewrite_asset_path(ref, assets_root)
                if local_path is None:
                    continue
                if not local_path.exists():
                    rel = str(local_path.relative_to(assets_root))
                    missing_by_ref[rel] += 1
                    missing.append(rel)
            if missing:
                missing_by_demo[demo_key] = sorted(set(missing))

    missing_dirs = Counter(str(Path(ref).parent) for ref in missing_by_ref)
    return {
        "hdf5": str(path),
        "demos_checked": checked,
        "demos_with_missing_assets": len(missing_by_demo),
        "missing_ref_count": int(sum(missing_by_ref.values())),
        "unique_missing_refs": len(missing_by_ref),
        "unique_missing_dirs": len(missing_dirs),
        "missing_dirs": missing_dirs.most_common(),
        "missing_refs": missing_by_ref.most_common(),
        "missing_by_demo": missing_by_demo,
        "parse_errors": parse_errors,
        "refs_by_demo": refs_by_demo,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_root", type=Path, required=True)
    parser.add_argument("--assets_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_demos_per_file", type=int, default=None)
    args = parser.parse_args()

    reports = []
    for hdf5_path in sorted(args.hdf5_root.glob("*.hdf5")):
        reports.append(audit_hdf5_file(hdf5_path, args.assets_root, args.max_demos_per_file))

    total = {
        "hdf5_files": len(reports),
        "demos_checked": sum(item["demos_checked"] for item in reports),
        "demos_with_missing_assets": sum(item["demos_with_missing_assets"] for item in reports),
        "unique_missing_refs": len({ref for item in reports for ref, _ in item["missing_refs"]}),
        "unique_missing_dirs": len({directory for item in reports for directory, _ in item["missing_dirs"]}),
    }
    missing_dirs = Counter()
    missing_refs = Counter()
    for item in reports:
        missing_dirs.update(dict(item["missing_dirs"]))
        missing_refs.update(dict(item["missing_refs"]))
    total["missing_dirs"] = missing_dirs.most_common()
    total["missing_refs"] = missing_refs.most_common()

    result = {"summary": total, "reports": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(total, indent=2, ensure_ascii=False))
    print(f"Wrote asset audit to {args.output}")


if __name__ == "__main__":
    main()
