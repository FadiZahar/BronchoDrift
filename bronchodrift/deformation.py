"""
Synthetic CT-to-body divergence (CTBD) on an airway centerline.

Convention: 
Input = pre-op CT (end-inspiration, no atelectasis).
Output = intra-procedural state (expiratory recoil + dependent atelectasis).

Two effects, both built from one mechanism: contract a set of nodes toward
an anchor along the tree, scaled by tree distance and an (optional) per-node
diaphragm-gradient weight. Mesh vertices follow via Inverse Distance Weighting 
(IDW) skinning.

Calibration: amplitude=1.0 -> ~8% SI-extent peak motion at the periphery
(from literature: up to ~20-25 mm in a ~250 mm lung, which is 8-10%; 
we can deliberately push further so the demo reads clearly).

Project's role is a synthetic CTBD generator with ground truth, not
a physical lung model. Replacing this with a learned deformation prior
trained on real 4D-CTs is the obvious next step.
"""

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# topology helpers
# ---------------------------------------------------------------------------

def detect_si_axis(T):
    """Infer the SI axis from root-to-deep-leaves direction.
    Dominant component of (deep_centroid - root) identifies which world
    axis is SI and which sign is inferior."""
    root_pos = np.array(T.nodes[T.graph["root"]]["pos"])
    max_gen = max(T.nodes[n].get("generation", 0) for n in T.nodes)
    deep = [T.nodes[n]["pos"] for n in T.nodes
            if T.nodes[n].get("generation", 0) >= max(max_gen - 1, 1)]
    delta = np.mean(deep, axis=0) - root_pos
    axis = int(np.argmax(np.abs(delta)))
    sign = 1 if delta[axis] > 0 else -1
    return axis, sign


def detect_lateral_axis(T, si_axis):
    """Lateral (L-R) axis = non-SI axis with the largest leaf spread."""
    leaf_pts = np.array([T.nodes[n]["pos"] for n in T.nodes if T.out_degree(n) == 0])
    spreads = leaf_pts.max(axis=0) - leaf_pts.min(axis=0)
    other = [a for a in (0, 1, 2) if a != si_axis]
    return other[0] if spreads[other[0]] > spreads[other[1]] else other[1]


def si_extent(T, axis):
    """SI length of the centerline; characteristic mesh size for calibration."""
    coords = np.array([T.nodes[n]["pos"][axis] for n in T.nodes])
    return float(coords.max() - coords.min())


def find_carina(T):
    """First bifurcation below the root: walk down until a node has >=2 children."""
    n = T.graph["root"]
    while T.out_degree(n) == 1:
        n = next(iter(T.successors(n)))
    return n


def subtree_nodes(T, hub):
    """All descendants of `hub` including `hub` itself."""
    return {hub} | nx.descendants(T, hub)


def path_distances_from(T, anchor, scope=None):
    """Tree-path distance from `anchor` to every node in `scope`."""
    U = T.to_undirected()
    if scope is not None:
        U = U.subgraph(scope).copy()
    return nx.single_source_dijkstra_path_length(U, anchor, weight="length")


def lung_side_of(T, node, lateral_axis):
    """+1 if the node sits on the +lateral side of the carina, -1 otherwise.
    Used to constrain atelectasis to one lung."""
    carina_lat = T.nodes[find_carina(T)]["pos"][lateral_axis]
    return 1 if T.nodes[node]["pos"][lateral_axis] >= carina_lat else -1


# ---------------------------------------------------------------------------
# per-node modulator weights (diaphragm gradient)
# ---------------------------------------------------------------------------

def respiration_weights(T, si_axis, si_sign, gen_exp=1.5, si_exp=2.0):
    """Per-node weight in [0,1]: 0.5 * (w_gen^p + w_si^q).

    w_gen: linear normalization of generation (0 at trachea, 1 at deepest).
      Generation is a proxy for mechanical freedom: The trachea is ~fixed.
      Empirically, motion grows with branching depth.

    w_si: linear normalization of SI position (0 at apex, 1 at diaphragm).
      Diaphragm motion is the dominant driver of lung deformation. 4D-CT
      studies show motion magnitude scaling roughly linearly with distance
      from the apex (Seppenwoolde 2002, Liu 2007).

    Summed (OR-gate) rather than multiplied (AND-gate): real upper-lobe
    peripheral airways move even though they're high in the lung, which AND 
    would suppress to ~0. Summation better matches the data.

    Exponents (default p=1.5, q=2): Quantitatively closer nodes nearly still 
    while the (farthest) periphery ones dominates. Specific p and q values are 
    more aesthetic. Exposed as parameters so the interactive tool can tune them.
    """
    nodes = list(T.nodes)
    gens = np.array([T.nodes[n].get("generation", 0) for n in nodes], dtype=float)
    si = np.array([T.nodes[n]["pos"][si_axis] for n in nodes], dtype=float)

    w_gen = gens / max(gens.max(), 1.0)
    span = si.max() - si.min()
    if span < 1e-9:
        w_si = np.ones_like(si)
    elif si_sign > 0:
        w_si = (si - si.min()) / span
    else:
        w_si = 1.0 - (si - si.min()) / span

    w = 0.5 * ((w_gen ** gen_exp) + (w_si ** si_exp))
    return {n: float(w[i]) for i, n in enumerate(nodes)}


# ---------------------------------------------------------------------------
# generic contraction toward an anchor
# ---------------------------------------------------------------------------

def contract_toward_anchor(T, anchor, scope, strength,
                           falloff_exp=1.0, weights=None):
    """Per-node displacement toward `anchor` along the tree.

    disp_i = strength * w_i * (d_i / d_max)^falloff_exp * (anchor_pos - x_i)

    d_i: is tree-path distance from anchor to node i. The anchor itself
      doesn't move (d=0). Nodes outside `scope` get zero displacement.
      `weights` (optional) scales magnitude per node -- pass respiration_weights 
      to add the diaphragm gradient on top of the topology term.

    falloff_exp < 1 (e.g. 0.5) gives mid-tree nodes a bigger share of the
      motion (sqrt curve); falloff_exp = 1 is linear; > 1 concentrates motion
      at the periphery only.
    """
    nodes = list(T.nodes)
    pos = np.array([T.nodes[n]["pos"] for n in nodes])
    anchor_pos = np.array(T.nodes[anchor]["pos"])

    dists = path_distances_from(T, anchor, scope=scope)
    if not dists:
        return np.zeros_like(pos)
    d_max = max(dists.values()) or 1.0

    disp = np.zeros_like(pos)
    for i, n in enumerate(nodes):
        if n not in dists:
            continue
        f = (dists[n] / d_max) ** falloff_exp
        w = weights[n] if weights is not None else 1.0
        disp[i] = strength * w * f * (anchor_pos - pos[i])
    return disp


def contract_radius(T, anchor, scope, max_shrink,
                    falloff_exp=1.0, weights=None):
    """Per-node lumen scale factor mirroring `contract_toward_anchor`.

    s_i = 1 - max_shrink * w_i * (d_i / d_max)^falloff_exp in scope, else 1.

    max_shrink calibration: airway lumen area drops 30-50% from TLC to FRC
    in small airways, which is ~15-30% radius reduction. Default 0.25 sits 
    at the midpoint.
    """
    nodes = list(T.nodes)
    s = np.ones(len(nodes))
    dists = path_distances_from(T, anchor, scope=scope)
    if not dists:
        return s
    d_max = max(dists.values()) or 1.0
    for i, n in enumerate(nodes):
        if n in dists:
            w = weights[n] if weights is not None else 1.0
            s[i] = 1.0 - max_shrink * w * (dists[n] / d_max) ** falloff_exp
    return s


# ---------------------------------------------------------------------------
# atelectasis scope: ipsilateral lung, at or below the hub's SI level
# ---------------------------------------------------------------------------

def atelectasis_scope(T, hub, si_axis, si_sign, lateral_axis):
    """Nodes ipsilateral to `hub` AND at or below its SI level. 
    
    Models a real lobar collapse: drags everything ipsilateral and inferior, 
    not just the airways downstream of the hub branch.
    """
    hub_side = lung_side_of(T, hub, lateral_axis)
    hub_si = T.nodes[hub]["pos"][si_axis]

    scope = set()
    for n in T.nodes:
        if lung_side_of(T, n, lateral_axis) != hub_side:
            continue
        # "at or below" the hub in the inferior direction
        node_si = T.nodes[n]["pos"][si_axis]
        if si_sign > 0 and node_si < hub_si:
            continue
        if si_sign < 0 and node_si > hub_si:
            continue
        scope.add(n)
    scope.add(hub)
    return scope


# ---------------------------------------------------------------------------
# Laplacian smoothing of fields on the centerline
# ---------------------------------------------------------------------------

def smooth_field(T, field, iters=3, alpha=0.5):
    """Average each node's field with its tree neighbors. Removes per-node
    discontinuities at bifurcations. Works on (N,) and (N,3) arrays."""
    if iters <= 0:
        return field
    U = T.to_undirected()
    nodes = list(T.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    out = field.copy()
    for _ in range(iters):
        new = out.copy()
        for n in nodes:
            nbrs = list(U.neighbors(n))
            if not nbrs:
                continue
            avg = np.mean([out[idx[m]] for m in nbrs], axis=0)
            new[idx[n]] = (1 - alpha) * out[idx[n]] + alpha * avg
        out = new
    return out


# ---------------------------------------------------------------------------
# propagation to mesh
# ---------------------------------------------------------------------------

def propagate_to_mesh(mesh_vertices, centerline_pos, centerline_disp,
                      centerline_scale, k=4):
    """IDW skinning: each vertex follows its k=4 nearest centerline nodes,
    translating + locally contracting around the local anchor. 
    Centerline ≈ bones, mesh ≈ skin; topology unchanged, only vertex positions move.
    """
    tree = cKDTree(centerline_pos)
    dists, idx = tree.query(mesh_vertices, k=k)
    if k == 1:
        idx = idx[:, None]
        dists = dists[:, None]
    w = 1.0 / (dists + 1e-9)
    w = w / w.sum(axis=1, keepdims=True)
    c_local = np.sum(centerline_pos[idx] * w[:, :, None], axis=1)
    d_local = np.sum(centerline_disp[idx] * w[:, :, None], axis=1)
    s_local = np.sum(centerline_scale[idx] * w, axis=1)[:, None]
    return c_local + s_local * (mesh_vertices - c_local) + d_local


# ---------------------------------------------------------------------------
# centerline update
# ---------------------------------------------------------------------------

def apply_centerline_displacement(T, disp, scale):
    """Return a deformed copy of T: shift positions, scale radii, refresh edges."""
    T_def = T.copy()
    nodes = list(T_def.nodes)
    for i, n in enumerate(nodes):
        T_def.nodes[n]["pos"] = np.array(T_def.nodes[n]["pos"]) + disp[i]
        if "radius" in T_def.nodes[n]:
            T_def.nodes[n]["radius"] = float(T_def.nodes[n]["radius"]) * float(scale[i])
    for u, v in T_def.edges:
        T_def[u][v]["length"] = float(np.linalg.norm(
            T_def.nodes[u]["pos"] - T_def.nodes[v]["pos"]))
    return T_def


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def deform(T, mesh, phi=1.0, amplitude=1.0,
           atel_hub=None, atel_strength=0.5, atel_radius_shrink=0.5,
           resp_base_strength=0.08, resp_max_shrink=0.25,
           resp_falloff_exp=0.5, atel_falloff_exp=1.0,
           smooth_iters=3, mode="drift",
           use_weights=True, gen_exp=1.5, si_exp=2.0,
           apply_translation=True, apply_radius=True):
    """Full pipeline: centerline + mesh -> deformed centerline + deformed mesh.

    Respiration: contract everything below the carina toward the carina,
      strength = phi * amplitude * resp_base_strength,
      modulated by tree distance and (if use_weights) the diaphragm gradient.

    Atelectasis: contract the ipsilateral lower-lung region toward atel_hub,
      with strength `atel_strength`. Scope is "same lung side, at or below
      the hub's SI level", not just the hub's airway subtree.

    Both fields are Laplacian-smoothed before mesh propagation.

    Apply_translation / apply_radius let you isolate the two effects:
      - apply_translation=False -> mesh keeps original positions, only lumens narrow
      - apply_radius=False -> mesh translates but lumens stay original size
    Useful for showing radius narrowing (bronchiolar thinning) in isolation.

    Returns (T_def, mesh_def, displacement_dict).
    """
    si_axis, si_sign = detect_si_axis(T)
    lateral_axis = detect_lateral_axis(T, si_axis)
    nodes = list(T.nodes)
    centerline_pos = np.array([T.nodes[n]["pos"] for n in nodes])

    weights = (respiration_weights(T, si_axis, si_sign, gen_exp, si_exp)
               if use_weights else None)

    # --- respiration ---
    carina = find_carina(T)
    resp_scope = subtree_nodes(T, carina)

    if mode == "drift":
        alpha = phi
    elif mode == "tidal":
        alpha = np.sin(np.pi * phi)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    resp_strength = alpha * amplitude * resp_base_strength
    u_resp = contract_toward_anchor(T, carina, resp_scope, resp_strength,
                                    falloff_exp=resp_falloff_exp, weights=weights)
    s_resp = contract_radius(T, carina, resp_scope,
                             max_shrink=alpha * amplitude * resp_max_shrink,
                             falloff_exp=resp_falloff_exp, weights=weights)

    # --- atelectasis: ipsilateral, at or below the hub's SI level ---
    if atel_hub is not None:
        scope = atelectasis_scope(T, atel_hub, si_axis, si_sign, lateral_axis)
        u_atel = contract_toward_anchor(T, atel_hub, scope, atel_strength,
                                        falloff_exp=atel_falloff_exp)
        s_atel = contract_radius(T, atel_hub, scope, max_shrink=atel_radius_shrink,
                                 falloff_exp=atel_falloff_exp)
    else:
        u_atel = np.zeros_like(u_resp)
        s_atel = np.ones(len(nodes))

    u_total = u_resp + u_atel
    s_total = s_resp * s_atel

    # --- gate the two effects independently ---
    if not apply_translation:
        u_total = np.zeros_like(u_total)
    if not apply_radius:
        s_total = np.ones_like(s_total)

    u_total = smooth_field(T, u_total, iters=smooth_iters)
    s_total = smooth_field(T, s_total, iters=smooth_iters)

    mesh_def = mesh.copy()
    mesh_def.vertices = propagate_to_mesh(
        mesh.vertices, centerline_pos, u_total, s_total, k=4)

    T_def = apply_centerline_displacement(T, u_total, s_total)
    displacement_dict = {n: u_total[i] for i, n in enumerate(nodes)}
    return T_def, mesh_def, displacement_dict