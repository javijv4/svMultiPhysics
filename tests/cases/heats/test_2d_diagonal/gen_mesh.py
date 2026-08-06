#!/usr/bin/env python3
"""
Generate a 2D square tissue mesh and a diagonal 1D line fiber mesh for
svMultiPhysics heatS MPC / EndNodes tests.

Coupling: three equally spaced interior nodes on the diagonal line
(at parametric s = 0.25, 0.5, 0.75), embedded in the tissue mesh.

Outputs under mesh/:
  volume.vtu, domain.dat
  mesh-surfaces/{left,right,bottom,top}.vtp
  domain.vtp is not used (TRI3 faces are invalid for 2D gnnb).
  coupling.vtp           (line edges around coupling nodes; EndNodes face)
  line.vtu, line.dat, line_nodes.txt
"""

from pathlib import Path

import gmsh
import numpy as np
import pyvista as pv


def edge_key(a, b):
    return (a, b) if a < b else (b, a)


def build_surface_vtp(points, edges, volume_node_ids, owner_element_ids):
    local_ids = {}
    local_points = []
    global_node_ids = []
    for edge in edges:
        for nid in edge:
            if nid not in local_ids:
                local_ids[nid] = len(local_points)
                local_points.append(points[nid])
                global_node_ids.append(volume_node_ids[nid])

    faces = np.empty(len(edges) * 3, dtype=np.int64)
    for i, (n0, n1) in enumerate(edges):
        faces[3 * i : 3 * i + 3] = (2, local_ids[n0], local_ids[n1])

    surface = pv.PolyData(np.asarray(local_points, dtype=float), faces=faces)
    surface.point_data["GlobalNodeID"] = np.asarray(global_node_ids, dtype=np.int32)
    surface.cell_data["GlobalElementID"] = np.asarray(owner_element_ids, dtype=np.int32)
    return surface




def build_coupling_vtp(points, tri_cells, volume_node_ids, volume_element_ids, coupling_xyz, tol=1e-10):
    """
    2D EndNodes face: line edges of all triangles incident to coupling points.
    TRI3 faces are invalid for nsd=2 (gnnb expects LIN faces).
    """
    tissue_xy = points[:, :2]
    couple_local = []
    for xy in coupling_xyz:
        d = np.linalg.norm(tissue_xy - np.asarray(xy)[:2], axis=1)
        couple_local.append(int(np.argmin(d)))
    couple_local = set(couple_local)

    edge_owner = {}
    for e_id, tri in enumerate(tri_cells):
        if not any(int(n) in couple_local for n in tri):
            continue
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = edge_key(int(a), int(b))
            if key not in edge_owner:
                edge_owner[key] = int(volume_element_ids[e_id])

    if len(edge_owner) == 0:
        raise RuntimeError("No coupling edges found around embedded coupling nodes")

    edges = list(edge_owner.keys())
    owners = [edge_owner[e] for e in edges]
    return build_surface_vtp(points, edges, volume_node_ids, owners)

def classify_square_curve(xmin, ymin, xmax, ymax, L, tol):
    xc = 0.5 * (xmin + xmax)
    yc = 0.5 * (ymin + ymax)
    dx = xmax - xmin
    dy = ymax - ymin
    if dx <= tol and dy > tol:
        if abs(xc) <= tol:
            return "left"
        if abs(xc - L) <= tol:
            return "right"
    if dy <= tol and dx > tol:
        if abs(yc) <= tol:
            return "bottom"
        if abs(yc - L) <= tol:
            return "top"
    return None


def extract_square_mesh(curve_phys_ids, volume_phys_id):
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    coords = np.asarray(node_coords, dtype=float).reshape(-1, 3)
    tag_to_local = {int(tag): i for i, tag in enumerate(node_tags)}
    points = coords.copy()

    tri_cells = []
    domain = []
    for dim, tag in gmsh.model.getEntities(2):
        phys = gmsh.model.getPhysicalGroupsForEntity(dim, tag)
        if len(phys) == 0 or int(phys[0]) != volume_phys_id:
            continue
        etypes, _, etag_conn = gmsh.model.mesh.getElements(dim, tag)
        for etype, conn in zip(etypes, etag_conn):
            name, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(etype)
            conn = np.asarray(conn, dtype=np.int64).reshape(-1, num_nodes)
            if num_nodes != 3:
                raise RuntimeError(f"Only triangular meshes are supported, got {name}")
            for tri in conn:
                tri_cells.append([tag_to_local[int(n)] for n in tri])
                domain.append(volume_phys_id)

    if len(tri_cells) == 0:
        raise RuntimeError("No triangular volume elements were extracted")

    tri_cells = np.asarray(tri_cells, dtype=np.int64)
    domain = np.asarray(domain, dtype=np.int32)
    volume_element_ids = np.arange(1, tri_cells.shape[0] + 1, dtype=np.int32)

    edge_to_element = {}
    for e_id, tri in zip(volume_element_ids, tri_cells):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_to_element[edge_key(int(a), int(b))] = int(e_id)

    used = np.unique(tri_cells.ravel())
    old_to_new = -np.ones(points.shape[0], dtype=np.int64)
    old_to_new[used] = np.arange(used.size, dtype=np.int64)
    points = points[used]
    tri_cells = old_to_new[tri_cells]

    edge_to_element_new = {}
    for (a, b), e_id in edge_to_element.items():
        if old_to_new[a] < 0 or old_to_new[b] < 0:
            continue
        edge_to_element_new[edge_key(int(old_to_new[a]), int(old_to_new[b]))] = e_id
    edge_to_element = edge_to_element_new

    surface_edges = {name: [] for name in curve_phys_ids}
    surface_owners = {name: [] for name in curve_phys_ids}
    for name, pid in curve_phys_ids.items():
        for curve_tag in gmsh.model.getEntitiesForPhysicalGroup(1, pid):
            etypes, _, etag_conn = gmsh.model.mesh.getElements(1, int(curve_tag))
            for etype, conn in zip(etypes, etag_conn):
                _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(etype)
                conn = np.asarray(conn, dtype=np.int64).reshape(-1, num_nodes)
                for line in conn:
                    a = tag_to_local[int(line[0])]
                    b = tag_to_local[int(line[1])]
                    if old_to_new[a] < 0 or old_to_new[b] < 0:
                        continue
                    n0 = int(old_to_new[a])
                    n1 = int(old_to_new[b])
                    owner = edge_to_element.get(edge_key(n0, n1))
                    if owner is None:
                        continue
                    surface_edges[name].append((n0, n1))
                    surface_owners[name].append(owner)

    return points, tri_cells, domain, surface_edges, surface_owners


def mesh_square(L, mesh_size, embed_points=None, model_name="square_heat"):
    gmsh.initialize()
    gmsh.model.add(model_name)
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)

    gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, L)
    embed_tags = []
    if embed_points is not None and len(embed_points) > 0:
        for xy in embed_points:
            tag = gmsh.model.occ.addPoint(float(xy[0]), float(xy[1]), 0.0, mesh_size)
            embed_tags.append(tag)
    gmsh.model.occ.synchronize()

    if embed_tags:
        gmsh.model.mesh.embed(0, embed_tags, 2, 1)

    volume_phys_id = 1
    gmsh.model.addPhysicalGroup(2, [1], tag=volume_phys_id)
    gmsh.model.setPhysicalName(2, volume_phys_id, "tissue")

    tol = 1e-4 * max(L, 1.0)
    boundary_curves = {name: [] for name in ("left", "right", "bottom", "top")}
    for dim, tag in gmsh.model.getEntities(1):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(dim, tag)
        name = classify_square_curve(xmin, ymin, xmax, ymax, L, tol)
        if name is not None:
            boundary_curves[name].append(tag)

    curve_phys_ids = {"left": 11, "right": 12, "bottom": 13, "top": 14}
    for name, pid in curve_phys_ids.items():
        tags = boundary_curves[name]
        if not tags:
            gmsh.finalize()
            raise RuntimeError(f"No boundary curves found for '{name}'")
        gmsh.model.addPhysicalGroup(1, tags, tag=pid)
        gmsh.model.setPhysicalName(1, pid, name)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.model.mesh.generate(2)

    points, tri_cells, domain, surface_edges, surface_owners = extract_square_mesh(
        curve_phys_ids, volume_phys_id
    )
    gmsh.finalize()
    return points, tri_cells, domain, surface_edges, surface_owners


def build_diagonal_line(L, n_line_nodes, coupling_fractions, n_stim_segments=1):
    """
    Build line from (0,0) to (L,L) with n_line_nodes equally spaced nodes.
    Coupling nodes are the line nodes nearest to the given parametric fractions.
    Domain 3 on the first n_stim_segments elements (left-bottom heat source);
    domain 2 elsewhere.
    """
    if n_line_nodes < 3:
        raise ValueError("n_line_nodes must be >= 3")

    s = np.linspace(0.0, 1.0, n_line_nodes)
    points = np.column_stack([s * L, s * L, np.zeros(n_line_nodes)])
    lines = np.column_stack(
        [np.arange(n_line_nodes - 1), np.arange(1, n_line_nodes)]
    ).astype(np.int64)

    domain = np.full(lines.shape[0], 2, dtype=np.int32)
    domain[: min(n_stim_segments, lines.shape[0])] = 3

    coupling_ids = []
    for frac in coupling_fractions:
        coupling_ids.append(int(np.argmin(np.abs(s - frac))))
    # Keep unique, sorted, and exclude pure endpoints if they were selected
    coupling_ids = sorted(set(coupling_ids))
    if len(coupling_ids) != len(coupling_fractions):
        # Fall back to exact index spacing for equally separated interior nodes
        step = (n_line_nodes - 1) // (len(coupling_fractions) + 1)
        coupling_ids = [step * (i + 1) for i in range(len(coupling_fractions))]

    return points, lines, domain, coupling_ids


def write_fiber_mesh(output_dir, points, lines, domain, coupling_ids):
    n_cells = lines.shape[0]
    cells = np.hstack([np.full((n_cells, 1), 2, dtype=np.int64), lines]).ravel()
    celltypes = np.full(n_cells, pv.CellType.LINE, dtype=np.uint8)
    fiber = pv.UnstructuredGrid(cells, celltypes, points)
    fiber.point_data["GlobalNodeID"] = np.arange(1, points.shape[0] + 1, dtype=np.int32)
    fiber.cell_data["GlobalElementID"] = np.arange(1, n_cells + 1, dtype=np.int32)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vtu_path = output_dir / "line.vtu"
    dat_path = output_dir / "line.dat"
    nodes_path = output_dir / "line_nodes.txt"
    fiber.save(str(vtu_path))
    np.savetxt(dat_path, domain, fmt="%d")
    nodes_1based = np.asarray(coupling_ids, dtype=np.int32) + 1
    np.savetxt(nodes_path, nodes_1based.reshape(1, -1), fmt="%d")
    return vtu_path, dat_path, nodes_path, nodes_1based


def write_tissue_mesh(
    output_dir,
    points,
    tri_cells,
    domain,
    surface_edges,
    surface_owners,
    coupling_xyz=None,
):
    n_cells = tri_cells.shape[0]
    volume_node_ids = np.arange(1, len(points) + 1, dtype=np.int32)
    volume_element_ids = np.arange(1, n_cells + 1, dtype=np.int32)

    cells = np.hstack(
        [np.full((n_cells, 1), 3, dtype=np.int64), tri_cells]
    ).ravel()
    celltypes = np.full(n_cells, pv.CellType.TRIANGLE, dtype=np.uint8)
    volume = pv.UnstructuredGrid(cells, celltypes, points)
    volume.point_data["GlobalNodeID"] = volume_node_ids
    volume.cell_data["GlobalElementID"] = volume_element_ids
    volume.cell_data["DomainID"] = domain

    output_dir = Path(output_dir)
    surfaces_dir = output_dir / "mesh-surfaces"
    output_dir.mkdir(parents=True, exist_ok=True)
    surfaces_dir.mkdir(parents=True, exist_ok=True)

    volume_path = output_dir / "volume.vtu"
    domain_path = output_dir / "domain.dat"
    volume.save(str(volume_path))
    np.savetxt(domain_path, domain, fmt="%d")

    for name in ("left", "right", "bottom", "top"):
        if name not in surface_edges or len(surface_edges[name]) == 0:
            raise RuntimeError(f"No edges found for surface '{name}'")
        surf = build_surface_vtp(
            points, surface_edges[name], volume_node_ids, surface_owners[name]
        )
        surf.save(str(surfaces_dir / f"{name}.vtp"))

    if coupling_xyz is None:
        raise RuntimeError("coupling_xyz is required to build the EndNodes coupling face")
    coupling_face = build_coupling_vtp(
        points, tri_cells, volume_node_ids, volume_element_ids, coupling_xyz
    )
    coupling_vtp_path = output_dir / "coupling.vtp"
    coupling_face.save(str(coupling_vtp_path))

    print("svMultiPhysics mesh readiness:")
    print(
        f"- Volume .vtu: {volume_path}, points={volume.n_points}, cells={volume.n_cells}, "
        f"GlobalNodeID={volume.point_data['GlobalNodeID'].dtype}, "
        f"GlobalElementID={volume.cell_data['GlobalElementID'].dtype}"
    )
    print(
        f"- Coupling face .vtp: {coupling_vtp_path}, points={coupling_face.n_points}, "
        f"cells={coupling_face.n_cells}"
    )
    print(
        f"- Domains: {domain_path}, dtype={domain.dtype}, rows={domain.size}, "
        f"expected cells={volume.n_cells}"
    )
    return volume_path, domain_path, coupling_vtp_path


def generate_case(
    square_side,
    mesh_size,
    n_line_nodes=21,
    coupling_fractions=(0.25, 0.5, 0.75),
    n_stim_segments=1,
    output_dir=None,
    tissue_domain_id=1,
    keep_existing_line=False,
):
    """
    Generate tissue + line meshes.

    If keep_existing_line is True and line.vtu already exists in output_dir,
    reuse that 1D mesh (for mesh-convergence tissue refinements).
    """
    if square_side <= 0.0:
        raise ValueError("square_side must be > 0")
    if mesh_size <= 0.0:
        raise ValueError("mesh_size must be > 0")
    if n_line_nodes < 5:
        raise ValueError("n_line_nodes must be >= 5 for three interior coupling nodes")

    L = float(square_side)
    h = float(mesh_size)
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "mesh"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Geometry: L={L}, mesh_size={h}, n_line_nodes={n_line_nodes}")

    line_vtu = output_dir / "line.vtu"
    if keep_existing_line and line_vtu.exists():
        fiber = pv.read(str(line_vtu))
        fiber_pts = np.asarray(fiber.points, dtype=float)
        fiber_lines = []
        for i in range(fiber.n_cells):
            cell = fiber.get_cell(i)
            fiber_lines.append([int(cell.point_ids[0]), int(cell.point_ids[1])])
        fiber_lines = np.asarray(fiber_lines, dtype=np.int64)
        fiber_domain = np.loadtxt(output_dir / "line.dat", dtype=np.int32)
        coupling_ids = np.loadtxt(output_dir / "line_nodes.txt", dtype=int).ravel() - 1
        coupling_ids = [int(i) for i in coupling_ids]
        print(f"Reusing existing line mesh: {fiber_pts.shape[0]} nodes")
    else:
        fiber_pts, fiber_lines, fiber_domain, coupling_ids = build_diagonal_line(
            L, n_line_nodes, coupling_fractions, n_stim_segments=n_stim_segments
        )
        write_fiber_mesh(output_dir, fiber_pts, fiber_lines, fiber_domain, coupling_ids)

    coupling_xyz = fiber_pts[coupling_ids][:, :2]
    print(f"Coupling nodes (0-based): {coupling_ids}")
    print(f"Coupling XYZ:\n{coupling_xyz}")

    points, tri_cells, domain, surface_edges, surface_owners = mesh_square(
        L, h, embed_points=coupling_xyz, model_name="tissue_heat"
    )
    domain[:] = tissue_domain_id
    write_tissue_mesh(
        output_dir,
        points,
        tri_cells,
        domain,
        surface_edges,
        surface_owners,
        coupling_xyz=coupling_xyz,
    )

    if not (keep_existing_line and line_vtu.exists()):
        pass
    else:
        # Ensure node file / dat still present
        if not (output_dir / "line_nodes.txt").exists():
            write_fiber_mesh(output_dir, fiber_pts, fiber_lines, fiber_domain, coupling_ids)

    tissue_xy = points[:, :2]
    for i, xy in enumerate(coupling_xyz):
        d = np.min(np.linalg.norm(tissue_xy - xy, axis=1))
        print(f"Coupling {i} embed distance to tissue node: {d:.3e}")

    print("- Remaining issues: none")
    return {
        "n_tissue_points": points.shape[0],
        "n_tissue_cells": tri_cells.shape[0],
        "coupling_ids": coupling_ids,
        "mesh_size": h,
    }


# ---------------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------------
square_side = 1.0
mesh_size = 0.1
n_line_nodes = 21
coupling_fractions = (0.25, 0.5, 0.75)
n_stim_segments = 1

base_dir = Path(__file__).resolve().parent
mesh_dir = base_dir / "mesh"

# Set HEAT_GEN_MESH_SKIP_MAIN=1 before importing this module from run_convergence.
import os

if os.environ.get("HEAT_GEN_MESH_SKIP_MAIN") != "1":
    generate_case(
        square_side=square_side,
        mesh_size=mesh_size,
        n_line_nodes=n_line_nodes,
        coupling_fractions=coupling_fractions,
        n_stim_segments=n_stim_segments,
        output_dir=mesh_dir,
    )