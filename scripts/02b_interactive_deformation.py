"""
Interactive tuning + camera-view inspection for BronchoDrift deformation.
    pip install polyscope
    python3 -m scripts.02b_interactive_deformation \
        --graph data/processed/centerline.pkl \
        --mesh  data/raw/airways_filled.obj
"""
import argparse, pickle, copy, time
from pathlib import Path
import numpy as np
import trimesh as tm
import polyscope as ps
import polyscope.imgui as psim

from bronchodrift import deformation as bddf


def list_atel_hubs(T, target_gen=3):
    """One representative hub per branch at ~target_gen, sorted inferior-first."""
    si_axis, si_sign = bddf.detect_si_axis(T)
    gens_in_tree = {T.nodes[n].get("generation", 0) for n in T.nodes}
    if target_gen not in gens_in_tree:
        target_gen = min(gens_in_tree, key=lambda g: abs(g - target_gen))

    by_branch = {}
    for n in T.nodes:
        if T.nodes[n].get("generation", 0) != target_gen:
            continue
        bid = T.nodes[n].get("branch_id", n)
        arc = T.nodes[n].get("arclength", 0.0)
        if bid not in by_branch or arc < by_branch[bid][1]:
            by_branch[bid] = (n, arc)
    candidates = [n for (n, _) in by_branch.values()]

    def subtree_mean_si(n):
        sub = bddf.subtree_nodes(T, n)
        return np.mean([T.nodes[m]["pos"][si_axis] for m in sub])

    candidates.sort(key=lambda n: -si_sign * subtree_mean_si(n))
    return candidates


def pick_demo_path(T):
    leaves = [n for n in T.nodes if T.out_degree(n) == 0]
    leaves.sort(key=lambda n: -T.nodes[n].get("generation", 0))
    target = leaves[0]
    path = [target]
    while path[-1] != T.graph["root"]:
        preds = list(T.predecessors(path[-1]))
        if not preds:
            break
        path.append(preds[0])
    return list(reversed(path))


DEFAULT_STATE = {
    "amplitude": 1.0, "phi": 1.0,
    "resp_base_strength": 0.08, "resp_max_shrink": 0.25, "resp_falloff_exp": 0.5,
    "atel_on": True, "atel_hub_gen": 3, "atel_hub_idx": 0,
    "atel_strength": 0.4, "atel_radius_shrink": 0.5, "atel_falloff_exp": 1.0,
    "smooth_iters": 3, "use_weights": True, "gen_exp": 1.5, "si_exp": 2.0,
    "apply_translation": True, "apply_radius": True,
    "path_t": 0.5,
    "tidal_mode": False, "tidal_animate": False, "tidal_speed": 0.5,
    "show_advanced": False,
}

ATEL_GEN_OPTIONS = [2, 3, 4, 5]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", required=True, type=Path)
    p.add_argument("--mesh", required=True, type=Path)
    args = p.parse_args()

    with open(args.graph, "rb") as f:
        T = pickle.load(f)
    mesh = tm.load(str(args.mesh), force="mesh")

    si_axis, si_sign = bddf.detect_si_axis(T)
    demo_path = pick_demo_path(T)
    state = copy.deepcopy(DEFAULT_STATE)

    hub_lists = {g: list_atel_hubs(T, target_gen=g) for g in ATEL_GEN_OPTIONS}

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("y_up" if si_axis == 1 else ("z_up" if si_axis == 2 else "x_up"))

    ps_orig = ps.register_surface_mesh(
        "CT (original)", mesh.vertices, mesh.faces,
        color=(0.55, 0.6, 0.75), transparency=0.25)
    ps_def = ps.register_surface_mesh(
        "Body (deformed)", mesh.vertices.copy(), mesh.faces,
        color=(0.68, 0.23, 0.23))
    ps_orig.set_back_face_policy("identical")
    ps_def.set_back_face_policy("identical")

    cl_pts = np.array([T.nodes[n]["pos"] for n in T.nodes])
    ps_cl = ps.register_point_cloud("centerline (CT frame)", cl_pts, radius=0.001,
                                    color=(0.2, 0.2, 0.2))

    path_pts = np.array([T.nodes[n]["pos"] for n in demo_path])
    ps_path = ps.register_curve_network(
        "catheter path (body frame)",
        path_pts,
        np.array([[i, i+1] for i in range(len(path_pts)-1)]),
        radius=0.0025, color=(1.0, 0.9, 0.2))

    # Two catheter-tip markers: yellow = body frame (where it actually is),
    # blue = CT frame (where the tracker thinks it is). The yellow-to-blue
    # distance in 3D is the live tracker error.
    ps_tip_body = ps.register_point_cloud(
        "catheter tip (body frame)", path_pts[:1], radius=0.010, color=(1.0, 0.95, 0.2))
    ps_tip_ct = ps.register_point_cloud(
        "catheter tip (CT frame)", path_pts[:1], radius=0.010, color=(0.125, 0.25, 1.0))

    cache = {"T_def": T, "mesh_def": mesh, "path_def": path_pts.copy()}

    def recompute():
        if state["tidal_mode"] or not state["atel_on"]:
            hub = None
        else:
            hubs = hub_lists[state["atel_hub_gen"]]
            idx = min(state["atel_hub_idx"], len(hubs) - 1) if hubs else 0
            hub = hubs[idx] if hubs else None

        T_def, mesh_def, _ = bddf.deform(
            T, mesh,
            phi=state["phi"], amplitude=state["amplitude"],
            atel_hub=hub,
            atel_strength=state["atel_strength"],
            atel_radius_shrink=state["atel_radius_shrink"],
            resp_base_strength=state["resp_base_strength"],
            resp_max_shrink=state["resp_max_shrink"],
            resp_falloff_exp=state["resp_falloff_exp"],
            atel_falloff_exp=state["atel_falloff_exp"],
            smooth_iters=state["smooth_iters"],
            use_weights=state["use_weights"],
            gen_exp=state["gen_exp"],
            si_exp=state["si_exp"],
            apply_translation=state["apply_translation"],
            apply_radius=state["apply_radius"],
            mode="tidal" if state["tidal_mode"] else "drift",
        )
        ps_def.update_vertex_positions(mesh_def.vertices)
        path_def = np.array([T_def.nodes[n]["pos"] for n in demo_path])
        ps_path.update_node_positions(path_def)
        cache["T_def"] = T_def
        cache["mesh_def"] = mesh_def
        cache["path_def"] = path_def
        update_tip()

    def update_tip():
        t = state["path_t"]
        n = len(cache["path_def"])
        f = t * (n - 1)
        i = int(np.clip(np.floor(f), 0, n - 2))
        a = f - i
        ct_tip = (1 - a) * path_pts[i] + a * path_pts[i + 1]
        body_tip = (1 - a) * cache["path_def"][i] + a * cache["path_def"][i + 1]
        ps_tip_ct.update_point_positions(ct_tip[None, :])
        ps_tip_body.update_point_positions(body_tip[None, :])
        cache["ct_tip"] = ct_tip
        cache["body_tip"] = body_tip
        cache["tip_idx"] = i

    def look_from(tip, idx, frame):
        """Camera at `tip`, looking down local path tangent.
        Hides the other mesh + the catheter path/tips so the lumen reads cleanly."""
        path_arr = path_pts if frame == "ct" else cache["path_def"]
        if idx + 1 < len(path_arr):
            tangent = path_arr[idx + 1] - path_arr[idx]
        else:
            tangent = path_arr[idx] - path_arr[idx - 1]
        n = np.linalg.norm(tangent)
        if n < 1e-6:
            return
        tangent = tangent / n
        eye = tip - tangent * 0.5
        target = tip + tangent * 5.0

        # Show only the inspected mesh; hide everything else for a clean view.
        ps_orig.set_enabled(frame == "ct")
        ps_def.set_enabled(frame == "body")
        if frame == "ct":
            ps_orig.set_transparency(1.0)
        else:
            ps_def.set_transparency(1.0)
        ps_path.set_enabled(False)
        ps_tip_ct.set_enabled(False)
        ps_tip_body.set_enabled(False)
        ps_cl.set_enabled(False)
        ps.look_at(eye, target)

    def reset_view():
        ps_orig.set_enabled(True)
        ps_def.set_enabled(True)
        ps_orig.set_transparency(0.25)
        ps_def.set_transparency(1.0)
        ps_path.set_enabled(True)
        ps_tip_ct.set_enabled(True)
        ps_tip_body.set_enabled(True)
        ps_cl.set_enabled(True)
        ps.reset_camera_to_home_view()

    def reset_state():
        state.update(copy.deepcopy(DEFAULT_STATE))
        recompute()
        reset_view()

    def callback():
        changed = False

        if state["tidal_mode"] and state["tidal_animate"]:
            t = time.time() * state["tidal_speed"]
            state["phi"] = 0.5 + 0.5 * np.sin(2 * np.pi * t)
            changed = True

        psim.TextUnformatted("=== respiration ===")
        c, state["amplitude"] = psim.SliderFloat("amplitude", state["amplitude"], 0.0, 3.0); changed |= c
        c, state["phi"] = psim.SliderFloat("phi", state["phi"], 0.0, 1.0); changed |= c
        c, state["resp_base_strength"] = psim.SliderFloat("resp_base_strength", state["resp_base_strength"], 0.0, 0.3); changed |= c
        c, state["resp_falloff_exp"] = psim.SliderFloat("resp_falloff_exp", state["resp_falloff_exp"], 0.2, 2.0); changed |= c
        c, state["resp_max_shrink"] = psim.SliderFloat("resp_max_shrink", state["resp_max_shrink"], 0.0, 0.6); changed |= c

        # # Diaphragm weight: when unticked, also zero amplitude so the user
        # # sees a clean "no deformation" baseline rather than the unweighted
        # # topology-only mode that can look unintuitive.
        # c, new_use_weights = psim.Checkbox("diaphragm weight", state["use_weights"])
        # if c:
        #     state["use_weights"] = new_use_weights
        #     if not new_use_weights:
        #         state["amplitude"] = 0.0
        #     changed = True

        # # Decouple translation and radius narrowing -- toggle either off to
        # isolate the other effect (e.g. show bronchiolar thinning alone).
        c, state["apply_translation"] = psim.Checkbox("apply translation", state["apply_translation"]); changed |= c
        psim.SameLine()
        c, state["apply_radius"] = psim.Checkbox("apply radius narrowing", state["apply_radius"]); changed |= c

        psim.Separator()
        psim.TextUnformatted("=== tidal animation ===")
        c, state["tidal_mode"] = psim.Checkbox("tidal mode (oscillate, no atelectasis)", state["tidal_mode"]); changed |= c
        if state["tidal_mode"]:
            _, state["tidal_animate"] = psim.Checkbox("auto-animate phi", state["tidal_animate"])
            _, state["tidal_speed"] = psim.SliderFloat("speed (Hz)", state["tidal_speed"], 0.1, 2.0)

        psim.Separator()
        psim.TextUnformatted("=== atelectasis ===")
        c, state["atel_on"] = psim.Checkbox("atelectasis on", state["atel_on"]); changed |= c

        gen_idx = ATEL_GEN_OPTIONS.index(state["atel_hub_gen"])
        c, gen_idx = psim.Combo("hub generation", gen_idx, [str(g) for g in ATEL_GEN_OPTIONS])
        if c:
            state["atel_hub_gen"] = ATEL_GEN_OPTIONS[gen_idx]
            state["atel_hub_idx"] = 0
            changed = True

        hubs = hub_lists[state["atel_hub_gen"]]
        if hubs:
            n_hubs = len(hubs)
            c, state["atel_hub_idx"] = psim.SliderInt(
                f"hub (0..{n_hubs-1})",
                min(state["atel_hub_idx"], n_hubs - 1), 0, n_hubs - 1); changed |= c
            lat_axis = bddf.detect_lateral_axis(T, si_axis)
            side = "R" if bddf.lung_side_of(T, hubs[state["atel_hub_idx"]], lat_axis) > 0 else "L"
            psim.TextUnformatted(f"  selected hub side: {side}")

        c, state["atel_strength"] = psim.SliderFloat("atel_strength", state["atel_strength"], 0.0, 1.0); changed |= c
        c, state["atel_radius_shrink"] = psim.SliderFloat("atel_radius_shrink", state["atel_radius_shrink"], 0.0, 0.9); changed |= c

        psim.Separator()
        c, state["smooth_iters"] = psim.SliderInt("smooth iters", state["smooth_iters"], 0, 10); changed |= c

        psim.Separator()
        _, state["show_advanced"] = psim.Checkbox("show advanced", state["show_advanced"])
        if state["show_advanced"]:
            c, state["gen_exp"] = psim.SliderFloat("gen_exp (weight curve)", state["gen_exp"], 0.5, 4.0); changed |= c
            c, state["si_exp"] = psim.SliderFloat("si_exp (weight curve)", state["si_exp"], 0.5, 4.0); changed |= c

        psim.Separator()
        psim.TextUnformatted("=== catheter ===")
        tip_changed, state["path_t"] = psim.SliderFloat("position along path", state["path_t"], 0.0, 1.0)

        if changed:
            recompute()
        elif tip_changed:
            update_tip()

        psim.Separator()
        if "ct_tip" in cache:
            ct = cache["ct_tip"]; bd = cache["body_tip"]
            err = float(np.linalg.norm(ct - bd))
            psim.TextUnformatted(f"catheter tip CT (blue):    ({ct[0]:6.1f},{ct[1]:6.1f},{ct[2]:6.1f})")
            psim.TextUnformatted(f"catheter tip body (yellow):({bd[0]:6.1f},{bd[1]:6.1f},{bd[2]:6.1f})")
            psim.TextUnformatted(f"tracker error: {err:.2f} (mesh units)")

        if psim.Button("view CT from tip"):
            look_from(cache["ct_tip"], cache["tip_idx"], "ct")
        psim.SameLine()
        if psim.Button("view body from tip"):
            look_from(cache["body_tip"], cache["tip_idx"], "body")
        if psim.Button("reset view"):
            reset_view()
        psim.SameLine()
        if psim.Button("RESET ALL PARAMETERS"):
            reset_state()

    recompute()
    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()