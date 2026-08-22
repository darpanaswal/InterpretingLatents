"""
One-time migration: for every outputs/thoughts/<family>/<task>/thoughts_*.pt
already on disk (from extract_thoughts.py runs that predate the safetensors
migration), write the equivalent .safetensors file. extract_thoughts.py no
longer writes .pt at all, and every consumer (markovianity_test.py,
mean_ablation.py, variance_decomposition.py) only reads .safetensors now --
this backfills old extractions so they still work without re-running the
(expensive, GPU-hours) extraction itself.

Carries over the same metadata extract_thoughts.py now writes natively
(n_thoughts, model, model_family, data_path, split) when present in the
source .pt; "instance_indices" is dropped -- it was always trivially
range(N) and nothing ever read it back.

Does NOT delete the source .pt files; remove them yourself once you've
confirmed the .safetensors files are good.

Usage:
    python -m helpers.convert_thoughts_to_safetensors
    python -m helpers.convert_thoughts_to_safetensors --force
"""

import argparse
import torch
from pathlib import Path
from safetensors.torch import save_file as save_safetensors
from src.config import THOUGHTS

METADATA_KEYS = ("n_thoughts", "model", "model_family", "data_path", "split")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing .safetensors files too.")
    args = ap.parse_args()

    pt_paths = sorted(THOUGHTS.glob("*/*/thoughts_*.pt"))
    if not pt_paths:
        print(f"[INFO] no thoughts_*.pt files found under {THOUGHTS}")
        return

    converted = skipped = failed = 0
    for pt_path in pt_paths:
        st_path = pt_path.with_suffix(".safetensors")
        if st_path.exists() and not args.force:
            skipped += 1
            continue
        try:
            blob = torch.load(pt_path, map_location="cpu", weights_only=False)
            thoughts = blob["thoughts"].contiguous()
            metadata = {k: str(blob[k]) for k in METADATA_KEYS if k in blob}
            save_safetensors({"thoughts": thoughts}, str(st_path),
                             metadata=metadata)
            print(f"[OK] {pt_path} -> {st_path}  {tuple(thoughts.shape)}")
            converted += 1
        except Exception as e:
            print(f"[FAIL] {pt_path}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n[DONE] converted={converted} skipped(existing)={skipped} failed={failed}")
    if converted:
        print("[NOTE] source .pt files were left in place; "
              "delete them yourself once you've verified the migration.")


if __name__ == "__main__":
    main()
