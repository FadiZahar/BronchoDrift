"""
Apply synthetic CTBD deformation to the centerline + mesh.

Usage:
    python3 -m scripts.02_apply_deformation \
        --graph data/processed/centerline.pkl \
        --mesh  data/raw/airways_filled.obj \
        --out-graph data/processed/centerline_deformed.ply \
        --out-mesh  data/processed/airway_deformed.obj

amplitude=1.0 -> ~8% SI-extent peak motion at the periphery.
--atel-hub-gen controls collapse extent (smaller=larger collapse).
--atel-hub-index picks which hub at that generation (0=most inferior).
Use --list-atel-hubs to see all candidates before committing.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import trimesh as tm

from bronchodrift import centerline as bdcl
from bronchodrift import deformation as bddf


def list_atel_hubs(T, target_gen=3):
    """One representative hub per branch at ~target_gen, sorted by inferior-ness.

    Each unbranched bronchial segment has many centerline nodes sharing the
    same `branch_id` and `generation`. They are collapsed to one entry per
    branch (the proximal end of that branch) so the candidate list matches
    the anatomical bronchi a clinician would name."""
    si_axis, si_sign = bddf.detect_si_axis(T)

    # find generations actually present, fall back to nearest available
    gens_in_tree = {T.nodes[n].get("generation", 0) for n in T.nodes}
    if target_gen not in gens_in_tree:
        target_gen = min(gens_in_tree, key=lambda g: abs(g - target_gen))

    # group nodes at this generation by branch_id, pick the proximal node
    # (smallest arclength) of each group as the branch's representative
    by_branch = {}
    for n in T.nodes:
        if T.nodes[n].get("generation", 0) != target_gen:
            continue
        bid = T.nodes[n].get("branch_id", n)  # fall back to node id if missing
        arc = T.nodes[n].get("arclength", 0.0)
        if bid not in by_branch or arc < by_branch[bid][1]:
            by_branch[bid] = (n, arc)

    candidates = [n for (n, _) in by_branch.values()]

    def subtree_mean_si(n):
        sub = bddf.subtree_nodes(T, n)
        return np.mean([T.nodes[m]["pos"][si_axis] for m in sub])

    candidates.sort(key=lambda n: -si_sign * subtree_mean_si(n))
    return candidates


def pick_atel_hub(T, target_gen=3, index=0):
    """Pick the `index`-th hub at ~target_gen, sorted by inferior-ness."""
    candidates = list_atel_hubs(T, target_gen)
    return candidates[min(index, len(candidates) - 1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", required=True, type=Path)
    p.add_argument("--mesh", required=True, type=Path)
    p.add_argument("--out-graph", required=True, type=Path)
    p.add_argument("--out-mesh", required=True, type=Path)
    p.add_argument("--phi", type=float, default=1.0)
    p.add_argument("--amplitude", type=float, default=1.0)
    p.add_argument("--resp-max-shrink", type=float, default=0.25,
                   help="peripheral lumen narrowing from respiration (0-1)")
    p.add_argument("--resp-base-strength", type=float, default=0.08,
                   help="peak respiration contraction as fraction of tree distance")
    p.add_argument("--resp-falloff-exp", type=float, default=0.5,
                   help="<1 spreads motion to mid-tree, =1 linear, >1 periphery-only")
    p.add_argument("--atel-falloff-exp", type=float, default=1.0)
    p.add_argument("--atel-hub-gen", type=int, default=3,
                   help="generation of the atelectasis hub (smaller=larger collapse)")
    p.add_argument("--atel-hub-index", type=int, default=0,
                   help="which hub at the chosen generation (0=most inferior, "
                        "1=second-most, etc.)")
    p.add_argument("--list-atel-hubs", action="store_true",
                   help="print all candidate atelectasis hubs at --atel-hub-gen and exit")
    p.add_argument("--atel-strength", type=float, default=0.4,
                   help="contraction fraction toward the atel hub (0-1)")
    p.add_argument("--atel-radius-shrink", type=float, default=0.5,
                   help="lumen narrowing at the atel periphery (0-1)")
    p.add_argument("--no-atel", action="store_true")
    p.add_argument("--no-weights", action="store_true",
                   help="disable the diaphragm-gradient weight modulator")
    p.add_argument("--smooth-iters", type=int, default=3)
    p.add_argument("--mode", choices=["drift", "tidal"], default="drift")
    args = p.parse_args()

    with open(args.graph, "rb") as f:
        T = pickle.load(f)

    # --- list-and-exit mode: don't load the mesh, don't deform, just print ---
    if args.list_atel_hubs:
        candidates = list_atel_hubs(T, args.atel_hub_gen)
        si_axis_, si_sign_ = bddf.detect_si_axis(T)
        lat_axis_ = bddf.detect_lateral_axis(T, si_axis_)
        print(f"[deform] {len(candidates)} candidate hubs at gen {args.atel_hub_gen}:")
        for i, n in enumerate(candidates):
            side = "R" if bddf.lung_side_of(T, n, lat_axis_) > 0 else "L"
            scope = bddf.atelectasis_scope(T, n, si_axis_, si_sign_, lat_axis_)
            pos = T.nodes[n]["pos"]
            print(f"  [{i:2d}] node {n} side={side} "
                  f"pos=({pos[0]:6.1f},{pos[1]:6.1f},{pos[2]:6.1f}) "
                  f"affecting {len(scope)} nodes")
        return

    mesh = tm.load(str(args.mesh), force="mesh")

    si_axis, si_sign = bddf.detect_si_axis(T)
    print(f"[deform] SI axis={si_axis} sign={si_sign:+d} "
          f"si_extent={bddf.si_extent(T, si_axis):.2f}")
    print(f"[deform] carina node: {bddf.find_carina(T)}")

    hub = None if args.no_atel else pick_atel_hub(T, args.atel_hub_gen, args.atel_hub_index)
    if hub is not None:
        lat_axis_ = bddf.detect_lateral_axis(T, si_axis)
        scope = bddf.atelectasis_scope(T, hub, si_axis, si_sign, lat_axis_)
        side = "R" if bddf.lung_side_of(T, hub, lat_axis_) > 0 else "L"
        print(f"[deform] atel hub: node {hub} (gen {T.nodes[hub].get('generation', '?')}, "
              f"side={side}) affecting {len(scope)} of {T.number_of_nodes()} nodes")

    T_def, mesh_def, disp = bddf.deform(
        T, mesh,
        phi=args.phi, amplitude=args.amplitude,
        atel_hub=hub,
        atel_strength=args.atel_strength,
        atel_radius_shrink=args.atel_radius_shrink,
        resp_base_strength=args.resp_base_strength,
        resp_max_shrink=args.resp_max_shrink,
        resp_falloff_exp=args.resp_falloff_exp,
        atel_falloff_exp=args.atel_falloff_exp,
        smooth_iters=args.smooth_iters,
        mode=args.mode,
        use_weights=not args.no_weights,
    )

    mags = np.linalg.norm(np.array(list(disp.values())), axis=1)
    print(f"[deform] centerline displacement: "
          f"mean={mags.mean():.2f} max={mags.max():.2f} "
          f"({100*mags.max()/bddf.si_extent(T, si_axis):.1f}% of SI extent)")

    bdcl.export_ply(T_def, args.out_graph)
    args.out_mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh_def.export(str(args.out_mesh))
    print(f"saved {args.out_graph}")
    print(f"saved {args.out_mesh}")


if __name__ == "__main__":
    main()