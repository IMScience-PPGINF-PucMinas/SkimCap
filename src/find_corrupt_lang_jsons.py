#!/usr/bin/env python3
"""
Scan all lang feature JSON files and report which ones are corrupted.

Usage:
    python find_corrupt_lang_jsons.py /path/to/lang_feature_dir [--fix] [--limit N]
"""

import argparse
import json
import os
import re
import sys


def try_load(path: str):
    """Return (ok_strict, ok_lenient, error_msg)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except OSError as e:
        return False, False, f"OSError: {e}"

    first_error = None

    # Strategy 1: standard strict parse
    try:
        json.loads(raw)
        return True, True, None
    except json.JSONDecodeError as e:
        first_error = str(e)

    # Strategy 2: strip ASCII control chars (except \t \n \r) and retry
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw)
    try:
        json.loads(cleaned)
        return False, True, first_error
    except json.JSONDecodeError:
        return False, False, first_error


def fix_file(path: str):
    """Overwrite file with control-char-stripped version."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw)
    json.loads(cleaned)  # validate before overwriting
    with open(path, "w", encoding="utf-8") as f:
        f.write(cleaned)


def main():
    parser = argparse.ArgumentParser(description="Find corrupt lang JSON files.")
    parser.add_argument("lang_dir", help="Directory containing *.json lang feature files")
    parser.add_argument("--fix", action="store_true",
                        help="Auto-fix files that pass lenient parsing by stripping control chars")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after N corrupt files found (0 = no limit)")
    args = parser.parse_args()

    lang_dir = args.lang_dir
    if not os.path.isdir(lang_dir):
        print(f"[ERROR] Directory not found: {lang_dir}")
        sys.exit(1)

    files = sorted(f for f in os.listdir(lang_dir) if f.endswith(".json"))
    total = len(files)
    print(f"Scanning {total} JSON files in {lang_dir} ...\n")

    strict_fail = []
    truly_broken = []
    scanned = 0

    for fname in files:
        scanned += 1
        path = os.path.join(lang_dir, fname)
        ok_strict, ok_lenient, err = try_load(path)

        if ok_strict:
            continue

        if ok_lenient:
            strict_fail.append((fname, err))
            print(f"  [FIXABLE] {fname}\n    {err}")
            if args.fix:
                try:
                    fix_file(path)
                    print(f"    → Fixed.")
                except Exception as ex:
                    print(f"    → Fix FAILED: {ex}")
        else:
            truly_broken.append((fname, err))
            print(f"  [BROKEN]  {fname}\n    {err}")

        n_corrupt = len(strict_fail) + len(truly_broken)
        if args.limit and n_corrupt >= args.limit:
            print(f"\n[--limit {args.limit} reached, stopping early]")
            break

    print("\n" + "="*60)
    print(f"Files scanned   : {scanned} / {total}")
    print(f"Fixable (ctrl-char): {len(strict_fail)}")
    print(f"Truly broken    : {len(truly_broken)}")

    if truly_broken:
        print("\nTruly broken (must regenerate):")
        for fname, err in truly_broken:
            print(f"  {fname}")

    if strict_fail and not args.fix:
        print(f"\nRun with --fix to auto-repair the {len(strict_fail)} fixable file(s).")

    sys.exit(1 if truly_broken else 0)


if __name__ == "__main__":
    main()