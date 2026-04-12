"""
Centerline extraction for an airway mesh (filled for best results).
 
Approach: voxelize the mesh interior, thin it to a 1-voxel skeleton with
skimage, then build a networkx tree rooted at the trachea.
 
One knob: `pitch` (voxel size in mesh units).
"""
 
from pathlib import Path
import pickle
 
import networkx as nx
import numpy as np
import trimesh as tm
from skimage.morphology import skeletonize
 
 
def load_mesh(path):
    """Load an OBJ, keep the largest connected component, fix normals."""
    mesh = tm.load(str(path), force="mesh")
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda m: len(m.vertices))
       
    mesh.fix_normals()
    return mesh
 
 
def skeleton_points(mesh, pitch):
    """Voxelize, fill, skeletonize. Return (points_world, voxel_ijk)."""
    vox = mesh.voxelized(pitch=pitch).fill()
    skel = skeletonize(vox.matrix.astype(bool))
    ijk = np.argwhere(skel)
    if len(ijk) == 0:
        raise RuntimeError("skeletonization produced no points (try smaller pitch)")
    pts = tm.transformations.transform_points(ijk.astype(float), vox.transform)
    return pts, ijk
 
 
def build_graph(pts, ijk):
    """Connect skeleton voxels by 26-neighborhood (3x3 - 1) adjacency."""
    G = nx.Graph()
    for i, p in enumerate(pts):
        G.add_node(i, pos=p)
 
    index = {tuple(c): i for i, c in enumerate(ijk)}
    offsets = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
               if (a, b, c) != (0, 0, 0)]
 
    for i, c in enumerate(ijk):
        for off in offsets:
            j = index.get((c[0] + off[0], c[1] + off[1], c[2] + off[2]))
            if j is not None and j > i:
                d = float(np.linalg.norm(pts[i] - pts[j]))
                G.add_edge(i, j, length=d)
 
    # Largest component only.
    cc = max(nx.connected_components(G), key=len)
    return G.subgraph(cc).copy()
 
 
def root_at_trachea(G, root_hint=None):
    """Pick a root and return a directed tree.

    Root selection, in order of preference:
      1. closest leaf to `root_hint` if given
      2. leaf endpoint of the longest unbranched run in the graph
         (the trachea is the longest degree-2 chain ending in a leaf —
         carina at one end, tracheal opening at the other)
    """
    leaves = [n for n in G.nodes if G.degree[n] == 1] or list(G.nodes)

    if root_hint is not None:
        leaf_pts = np.array([G.nodes[n]["pos"] for n in leaves])
        root = leaves[int(np.argmin(np.linalg.norm(leaf_pts - root_hint, axis=1)))]
    else:
        root = _longest_run_leaf(G)

    # MST in case of voxel-skeleton loops.
    if not nx.is_tree(G):
        G = nx.minimum_spanning_tree(G, weight="length")

    T = nx.DiGraph()
    for n, data in G.nodes(data=True):
        T.add_node(n, **data)
    for parent, child in nx.bfs_edges(G, source=root):
        T.add_edge(parent, child, length=G[parent][child]["length"])
    T.graph["root"] = root
    return T


def _longest_run_leaf(G):
    """Return the leaf at the end of the longest unbranched run.

    Walks from every leaf along degree-2 nodes until it hits a bifurcation
    (degree >= 3). The leaf whose walk accumulates the most arclength wins.
    """
    leaves = [n for n in G.nodes if G.degree[n] == 1]
    best_leaf, best_len = leaves[0], -1.0

    for leaf in leaves:
        prev, curr = None, leaf
        total = 0.0
        while True:
            neighbors = [m for m in G.neighbors(curr) if m != prev]
            if len(neighbors) != 1:  # leaf again, or hit a bifurcation
                break
            nxt = neighbors[0]
            total += G[curr][nxt]["length"]
            prev, curr = curr, nxt
            if G.degree[curr] != 2:  # arrived at bifurcation or another leaf
                break
        if total > best_len:
            best_len, best_leaf = total, leaf

    return best_leaf
 
 
def smooth(T, iterations=5, alpha=0.5):
    """Laplacian smoothing of node positions.
 
    Each degree-2 (interior) node moves toward the average of its two neighbors.
    Leaves, bifurcations, and the root are held fixed so topology and endpoints
    don't drift.
    """
    U = T.to_undirected()
    fixed = {n for n in U.nodes if U.degree[n] != 2}
    fixed.add(T.graph["root"])
 
    for _ in range(iterations):
        new_pos = {}
        for n in U.nodes:
            if n in fixed:
                continue
            neighbors = list(U.neighbors(n))
            avg = np.mean([U.nodes[m]["pos"] for m in neighbors], axis=0)
            new_pos[n] = (1 - alpha) * U.nodes[n]["pos"] + alpha * avg
        for n, p in new_pos.items():
            T.nodes[n]["pos"] = p
            U.nodes[n]["pos"] = p
 
 
def annotate(T, mesh):
    """Add generation, branch_id, arclength, radius to each node."""
    root = T.graph["root"]
 
    T.nodes[root]["arclength"] = 0.0
    T.nodes[root]["generation"] = 0
    T.nodes[root]["branch_id"] = 0
    next_branch = 1
 
    for parent, child in nx.bfs_edges(T, source=root):
        T.nodes[child]["arclength"] = T.nodes[parent]["arclength"] + T[parent][child]["length"]
        bif = T.out_degree(parent) >= 2
        T.nodes[child]["generation"] = T.nodes[parent]["generation"] + (1 if bif else 0)
        if bif:
            T.nodes[child]["branch_id"] = next_branch
            next_branch += 1
        else:
            T.nodes[child]["branch_id"] = T.nodes[parent]["branch_id"]
 
    # Radius ~= distance to mesh surface.
    pts = np.array([T.nodes[n]["pos"] for n in T.nodes])
    _, dists, _ = tm.proximity.ProximityQuery(mesh).on_surface(pts)
    for n, d in zip(T.nodes, dists):
        T.nodes[n]["radius"] = float(d)
 
 
def extract(mesh_path, pitch=0.5, root_hint=None, smooth_iters=5):
    """Full pipeline: mesh path -> annotated rooted airway tree."""
    mesh = load_mesh(mesh_path)
    pts, ijk = skeleton_points(mesh, pitch=pitch)
    G = build_graph(pts, ijk)
    T = root_at_trachea(G, root_hint=root_hint)
    if smooth_iters > 0:
        smooth(T, iterations=smooth_iters)
        # Recompute edge lengths after smoothing moved the points.
        for u, v in T.edges:
            T[u][v]["length"] = float(np.linalg.norm(T.nodes[u]["pos"] - T.nodes[v]["pos"]))
    annotate(T, mesh)
    return T, mesh
 
 
# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
 
def save(T, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(T, f)
 
 
def export_ply(T, path):
    """Export as a PLY line-set, colored by generation."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nodes = list(T.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    pos = np.array([T.nodes[n]["pos"] for n in nodes])
    gen = np.array([T.nodes[n]["generation"] for n in nodes], dtype=float)
    g = gen / max(gen.max(), 1)
    rgb = np.stack([(255 * g), (255 * (1 - abs(g - 0.5) * 2)), (255 * (1 - g))], axis=1).astype(np.uint8)
    edges = [(idx[u], idx[v]) for u, v in T.edges]
 
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pos)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\nend_header\n")
        for p, c in zip(pos, rgb):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
        for a, b in edges:
            f.write(f"{a} {b}\n")
 
 
def summary(T):
    leaves = sum(1 for n in T.nodes if T.out_degree(n) == 0)
    bifs = sum(1 for n in T.nodes if T.out_degree(n) >= 2)
    max_gen = max(T.nodes[n]["generation"] for n in T.nodes)
    radii = [T.nodes[n]["radius"] for n in T.nodes]
    return (
        f"  nodes:        {T.number_of_nodes()}\n"
        f"  leaves:       {leaves}\n"
        f"  bifurcations: {bifs}\n"
        f"  max gen:      {max_gen}\n"
        f"  radius:       {min(radii):.2f} – {max(radii):.2f}\n"
    )