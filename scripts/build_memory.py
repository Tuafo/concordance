"""Build Concordance corpus memory files and on-demand LiveMemory index.

Examples:
    python scripts/build_memory.py --download              # fetch enwik8 and build train/val, test, and live indexes
    python scripts/build_memory.py --data data/enwik8      # build all indexes from an existing enwik8 file
    python scripts/build_memory.py --live-only --src my_corpus.txt --live-dir data/live_custom
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concordance.live import build_live_index
from concordance.memory import ORDERS, build_corpus_memory, build_test_memory

ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"


def download_enwik8(dest: str) -> None:
    if os.path.exists(dest):
        print(f"found existing {dest}", flush=True)
        return
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    zip_path = dest + ".zip"
    print(f"downloading {ENWIK8_URL} -> {zip_path} ...", flush=True)
    urllib.request.urlretrieve(ENWIK8_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("enwik8", os.path.dirname(dest) or ".")
    os.remove(zip_path)
    print(f"extracted {dest}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Concordance exact corpus memory and live hash index.")
    ap.add_argument("--data", default="data/enwik8", help="Path to raw byte corpus (default: data/enwik8)")
    ap.add_argument("--out-dir", default="data", help="Output directory for .npy memory files")
    ap.add_argument("--live-dir", default="data/live", help="Output directory for on-demand LiveMemory index")
    ap.add_argument("--download", action="store_true", help="Download and extract enwik8 if missing")
    ap.add_argument("--train-val-only", action="store_true", help="Only build train+val memory files")
    ap.add_argument("--test-only", action="store_true", help="Only build official 95M-100M test memory files")
    ap.add_argument("--live-only", action="store_true", help="Only build on-demand LiveMemory hash index")
    ap.add_argument("--src", default=None, help="Override source file for --live-only custom corpus indexing")
    ap.add_argument("--start", type=int, default=0, help="Start byte offset for --live-only")
    ap.add_argument("--end", type=int, default=90_000_000, help="End byte offset for --live-only (0 = EOF)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.download:
        download_enwik8(args.data)

    os.makedirs(args.out_dir, exist_ok=True)
    run_all = not (args.train_val_only or args.test_only or args.live_only)

    if run_all or args.train_val_only:
        build_corpus_memory(
            data_path=args.data,
            stream_out=os.path.join(args.out_dir, "corpus_mem2.npy"),
            val_frozen_out=os.path.join(args.out_dir, "corpus_mem2_val_frozen.npy"),
            orders=ORDERS,
            device=args.device,
        )

    if run_all or args.test_only:
        build_test_memory(
            data_path=args.data,
            stream_out=os.path.join(args.out_dir, "corpus_mem2_test.npy"),
            frozen_out=os.path.join(args.out_dir, "corpus_mem2_test_frozen.npy"),
            orders=ORDERS,
            device=args.device,
        )

    if run_all or args.live_only:
        src = args.src or args.data
        end = None if args.end == 0 else args.end
        build_live_index(src_path=src, out_dir=args.live_dir, start=args.start, end=end, orders=ORDERS)


if __name__ == "__main__":
    main()
