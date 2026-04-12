"""
Build the airway centerline graph from a mesh OBJ.

Usage:
    python3 -m scripts.01_build_centerline \
        --mesh data/raw/airways_filled.obj \
        --out  data/processed/centerline.pkl \
        --ply  data/processed/centerline.ply

Tune --pitch (voxel size in mesh units) if the centerline looks wrong:
    smaller = more detail, more memory.
"""

import argparse
from pathlib import Path

import numpy as np

from bronchodrift import centerline as bdcl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--ply", type=Path, default=None)
    p.add_argument("--pitch", type=float, default=0.6)
    p.add_argument("--root-hint", type=float, nargs=3, default=None)
    args = p.parse_args()

    hint = np.array(args.root_hint) if args.root_hint else None
    print(f"[centerline] {args.mesh} (pitch={args.pitch})")

    T, _ = bdcl.extract(args.mesh, pitch=args.pitch, root_hint=hint)
    print(bdcl.summary(T))

    bdcl.save(T, args.out)
    print(f"saved {args.out}")
    if args.ply:
        bdcl.export_ply(T, args.ply)
        print(f"saved {args.ply}")


if __name__ == "__main__":
    main()
