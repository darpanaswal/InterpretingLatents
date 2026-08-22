"""
Snapshot/restore GPT-2 tables and plots so re-running the plotting
scripts for llama can't silently clobber them.

Most plotting scripts here already skip writing a family they found no
raw data for (see plot_gradient_geometry.py, plot_markov.py,
plot_gradient_subspace_interventions.py, plot_causal_trace.py,
plot_remove_thoughts.py, plot_epiphenomena.py, plot_mean_ablation.py --
all check for discovered/present data before writing). One does not:
experiments.geometry.variance_decomposition's write_tex_tables() writes
unconditionally for every family in --model_family (or both, for
--model_family both), with no check that any data was actually found.
Since GPT-2's raw outputs/variance_decomposition/gpt2/ tree no longer
exists on disk, invoking it with --model_family gpt2 or both would
silently overwrite variance_decomposition_gpt2.tex with an empty/garbage
table.

Rather than rely on every script (and every future invocation) getting
family-handling right, this snapshots every currently-good *gpt2* file
under Tables/ and Plots/ before you regenerate anything for llama, and
restores them afterward -- so even a mistaken --model_family both, or a
script I didn't audit closely enough, can't cost you the GPT-2 outputs.

Usage:
    python -m helpers.protect_gpt2_outputs snapshot   # before regenerating
    ... run llama-only experiments + plotting scripts ...
    python -m helpers.protect_gpt2_outputs restore    # after
    python -m helpers.protect_gpt2_outputs check       # optional: report
                                                          what changed
                                                          without restoring
"""

import sys
import shutil
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WATCHED_DIRS = [BASE_DIR / "Tables", BASE_DIR / "Plots"]
BACKUP_DIR = BASE_DIR / ".gpt2_outputs_backup"


def _gpt2_files():
    """Every file under Tables/ or Plots/ with 'gpt2' in its name."""
    files = []
    for d in WATCHED_DIRS:
        if not d.exists():
            continue
        files.extend(p for p in d.rglob("*gpt2*") if p.is_file())
    return files


def snapshot():
    files = _gpt2_files()
    if not files:
        print(f"[WARN] no *gpt2* files found under {[str(d) for d in WATCHED_DIRS]}")
        return
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)
    BACKUP_DIR.mkdir(parents=True)
    for f in files:
        rel = f.relative_to(BASE_DIR)
        dest = BACKUP_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
    print(f"[OK] snapshotted {len(files)} file(s) -> {BACKUP_DIR}")


def _backup_files():
    if not BACKUP_DIR.exists():
        return []
    return [p for p in BACKUP_DIR.rglob("*") if p.is_file()]


def check():
    backups = _backup_files()
    if not backups:
        print(f"[WARN] no snapshot found at {BACKUP_DIR} -- run 'snapshot' first")
        return
    changed, missing = [], []
    for b in backups:
        rel = b.relative_to(BACKUP_DIR)
        live = BASE_DIR / rel
        if not live.exists():
            missing.append(rel)
        elif live.read_bytes() != b.read_bytes():
            changed.append(rel)
    if not changed and not missing:
        print(f"[OK] all {len(backups)} snapshotted file(s) unchanged")
        return
    for rel in missing:
        print(f"[MISSING] {rel}")
    for rel in changed:
        print(f"[CHANGED] {rel}")
    print(f"\n{len(missing)} missing, {len(changed)} changed out of {len(backups)}. "
          f"Run 'restore' to put the snapshotted versions back.")


def restore():
    backups = _backup_files()
    if not backups:
        print(f"[WARN] no snapshot found at {BACKUP_DIR} -- nothing to restore")
        return
    restored = 0
    for b in backups:
        rel = b.relative_to(BACKUP_DIR)
        live = BASE_DIR / rel
        if not live.exists() or live.read_bytes() != b.read_bytes():
            live.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(b, live)
            restored += 1
    print(f"[OK] restored {restored} file(s) (of {len(backups)} snapshotted) to their GPT-2 originals")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["snapshot", "check", "restore"])
    args = ap.parse_args()
    {"snapshot": snapshot, "check": check, "restore": restore}[args.action]()


if __name__ == "__main__":
    main()
