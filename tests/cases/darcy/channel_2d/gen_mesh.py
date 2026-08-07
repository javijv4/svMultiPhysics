#!/usr/bin/env python3
"""
Generate 2D square meshes with a horizontal channel for svMultiPhysics Darcy tests.

Geometry (z = 0):
  - Outer square: [0, L] x [0, L]
  - Channel: [0, channel_length_fraction * L] x [(L - w) / 2, (L + w) / 2]
    (opens on the left, ends at 3/4 of the side length by default)

Outputs:
  mesh/              filled channel + tissue domains (implicit)
    volume.vtu, domain.dat, mesh-surfaces/{left,right,bottom,top}.vtp

  explicit_mesh/     channel removed (notch); Dirichlet face on channel boundary
    volume.vtu, domain.dat, mesh-surfaces/{left,right,bottom,top,channel}.vtp

  mpc_mesh/          full square tissue + 1D channel centerline (MPC)
    volume.vtu, domain.dat, mesh-surfaces/{left,right,bottom,top}.vtp
    channel.vtu, channel.dat, channel_nodes.txt
"""

from pathlib import Path

import gmsh
import numpy as np
import pyvista as pv


def edge_key(a, b):
    return (a, b) if a < b else (b, a)


def build_surface_vtp(points, edges, volume_node_ids, owner_element_ids):
    """Build a 2-point polygon PolyData surface matching svMultiPhysics 2D faces."""
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


def classify_curve(xmin, ymin, xmax, ymax, L, Lc, y0, y1, tol, explicit):
    """Return boundary name for a curve bounding box, or None if internal."""
    xc = 0.5 * (xmin + xmax)
    yc = 0.5 * (ymin + ymax)
    dx = xmax - xmin
    dy = ymax - ymin
    is_vertical = dx <= tol and dy > tol
    is_horizontal = dy <= tol and dx > tol

    if is_vertical and abs(xc) <= tol:
        # Left side: for explicit, channel inlet belongs to the channel face.
        if explicit and y0 - tol <= yc <= y1 + tol:
            return "channel"
        return "left"
    if is_vertical and abs(xc - L) <= tol:
        return "right"
    if is_vertical and abs(xc - Lc) <= tol and y0 - tol <= yc <= y1 + tol:
        # Channel tip: outer boundary only when channel is removed.
        return "channel" if explicit else None

    if is_horizontal and abs(yc) <= tol:
        return "bottom"
    if is_horizontal and abs(yc - L) <= tol:
        return "top"
    if is_horizontal and abs(yc - y0) <= tol and xc <= Lc + tol:
        return "channel" if explicit else None
    if is_horizontal and abs(yc - y1) <= tol and xc <= Lc + tol:
        return "channel" if explicit else None

    return None


def write_sv_mesh(output_dir, points, tri_cells, domain, surface_edges, surface_owners, surface_names):
    """Write volume.vtu, domain.dat, and surface .vtp files."""
    n_cells = tri_cells.shape[0]
    volume_node_ids = np.arange(1, len(points) + 1, dtype=np.int32)
    volume_element_ids = np.arange(1, n_cells + 1, dtype=np.int32)

    cells = np.hstack(
        [
            np.full((n_cells, 1), 3, dtype=np.int64),
            tri_cells,
        ]
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

    surface_paths = {}
    for name in surface_names:
        if name not in surface_edges or len(surface_edges[name]) == 0:
            raise RuntimeError(f"No edges found for surface '{name}' in {output_dir}")
        surf = build_surface_vtp(
            points,
            surface_edges[name],
            volume_node_ids,
            surface_owners[name],
        )
        path = surfaces_dir / f"{name}.vtp"
        surf.save(str(path))
        surface_paths[name] = path

    print(f"svMultiPhysics mesh readiness ({output_dir.name}):")
    print(
        f"- Volume .vtu: {volume_path}, points={volume.n_points}, cells={volume.n_cells}, "
        f"GlobalNodeID={volume.point_data['GlobalNodeID'].dtype}, "
        f"GlobalElementID={volume.cell_data['GlobalElementID'].dtype}"
    )
    for name, path in surface_paths.items():
        surf = pv.read(str(path))
        print(
            f"- Surface .vtp [{name}]: {path}, points={surf.n_points}, cells={surf.n_cells}, "
            f"GlobalNodeID={surf.point_data['GlobalNodeID'].dtype}, "
            f"GlobalElementID={surf.cell_data['GlobalElementID'].dtype}"
        )
    print(
        f"- Domains: {domain_path}, dtype={domain.dtype}, rows={domain.size}, "
        f"expected cells={volume.n_cells}, unique={np.unique(domain)}"
    )
    return volume_path, domain_path, surface_paths


def extract_mesh_from_gmsh(physical_surface_ids, curve_phys_ids):
    """Extract points, triangles, domains, and boundary edges from the current gmsh model."""
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    coords = np.asarray(node_coords, dtype=float).reshape(-1, 3)

    tag_to_local = {int(tag): i for i, tag in enumerate(node_tags)}
    points = coords.copy()

    tri_cells = []
    domain = []
    for dim, tag in gmsh.model.getEntities(2):
        phys = gmsh.model.getPhysicalGroupsForEntity(dim, tag)
        if len(phys) == 0:
            continue
        domain_id = int(phys[0])
        if domain_id not in physical_surface_ids:
            continue
        etypes, _, etag_conn = gmsh.model.mesh.getElements(dim, tag)
        for etype, conn in zip(etypes, etag_conn):
            name, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(etype)
            conn = np.asarray(conn, dtype=np.int64).reshape(-1, num_nodes)
            if num_nodes != 3:
                raise RuntimeError(f"Only triangular meshes are supported, got {name}")
            for tri in conn:
                tri_cells.append([tag_to_local[int(n)] for n in tri])
                domain.append(domain_id)

    if len(tri_cells) == 0:
        raise RuntimeError("No triangular volume elements were extracted")

    tri_cells = np.asarray(tri_cells, dtype=np.int64)
    domain = np.asarray(domain, dtype=np.int32)
    volume_element_ids = np.arange(1, tri_cells.shape[0] + 1, dtype=np.int32)

    edge_to_element = {}
    for e_id, tri in zip(volume_element_ids, tri_cells):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_to_element[edge_key(int(a), int(b))] = int(e_id)

    # Keep only nodes used by retained volume elements, then compact.
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
        entities = gmsh.model.getEntitiesForPhysicalGroup(1, pid)
        for curve_tag in entities:
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


def build_geometry_and_mesh(include_channel_volume):
    """
    Build OCC geometry, mesh, and return extracted arrays.

    include_channel_volume=True  -> tissue + channel domains (implicit)
    include_channel_volume=False -> tissue only, channel cut out (explicit)
    """
    gmsh.initialize()
    gmsh.model.add("darcy_channel_2d")
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)

    if include_channel_volume:
        tissue_bot = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, y0)
        channel = gmsh.model.occ.addRectangle(0.0, y0, 0.0, Lc, w)
        tissue_mid = gmsh.model.occ.addRectangle(Lc, y0, 0.0, L - Lc, w)
        tissue_top = gmsh.model.occ.addRectangle(0.0, y1, 0.0, L, L - y1)
        ov, _ = gmsh.model.occ.fragment(
            [(2, tissue_bot), (2, channel), (2, tissue_mid), (2, tissue_top)],
            [],
        )
        gmsh.model.occ.synchronize()

        faces = []
        for dim, tag in ov:
            if dim != 2:
                continue
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(dim, tag)
            xc = 0.5 * (xmin + xmax)
            yc = 0.5 * (ymin + ymax)
            faces.append((tag, xc, yc))

        tissue_tags = []
        channel_tags = []
        for tag, xc, yc in faces:
            if y0 - tol <= yc <= y1 + tol and xc <= Lc + tol:
                channel_tags.append(tag)
            else:
                tissue_tags.append(tag)

        if len(channel_tags) != 1 or len(tissue_tags) != 3:
            gmsh.finalize()
            raise RuntimeError(
                f"Unexpected face split: channel={channel_tags}, tissue={tissue_tags}"
            )

        gmsh.model.addPhysicalGroup(2, tissue_tags, tag=tissue_domain_id)
        gmsh.model.setPhysicalName(2, tissue_domain_id, "tissue")
        gmsh.model.addPhysicalGroup(2, channel_tags, tag=channel_domain_id)
        gmsh.model.setPhysicalName(2, channel_domain_id, "channel")
        volume_phys_ids = {tissue_domain_id, channel_domain_id}
        surface_names = ["left", "right", "bottom", "top"]
    else:
        square = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, L)
        channel = gmsh.model.occ.addRectangle(0.0, y0, 0.0, Lc, w)
        ov, _ = gmsh.model.occ.cut([(2, square)], [(2, channel)], removeTool=True)
        gmsh.model.occ.synchronize()

        tissue_tags = [tag for dim, tag in ov if dim == 2]
        if len(tissue_tags) == 0:
            gmsh.finalize()
            raise RuntimeError("Explicit cut produced no tissue faces")

        gmsh.model.addPhysicalGroup(2, tissue_tags, tag=tissue_domain_id)
        gmsh.model.setPhysicalName(2, tissue_domain_id, "tissue")
        volume_phys_ids = {tissue_domain_id}
        surface_names = ["left", "right", "bottom", "top", "channel"]

    # Boundary curve physical groups
    explicit = not include_channel_volume
    boundary_curves = {name: [] for name in ("left", "right", "bottom", "top", "channel")}
    for dim, tag in gmsh.model.getEntities(1):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(dim, tag)
        name = classify_curve(xmin, ymin, xmax, ymax, L, Lc, y0, y1, tol, explicit)
        if name is not None:
            boundary_curves[name].append(tag)

    curve_phys_ids = {"left": 11, "right": 12, "bottom": 13, "top": 14, "channel": 15}
    active_curve_phys = {}
    for name in surface_names:
        tags = boundary_curves[name]
        if not tags:
            gmsh.finalize()
            raise RuntimeError(f"No boundary curves found for '{name}'")
        pid = curve_phys_ids[name]
        gmsh.model.addPhysicalGroup(1, tags, tag=pid)
        gmsh.model.setPhysicalName(1, pid, name)
        active_curve_phys[name] = pid

    # Mesh sizes: refine near the channel region
    gmsh.model.mesh.field.add("Box", 1)
    gmsh.model.mesh.field.setNumber(1, "VIn", channel_mesh_size)
    gmsh.model.mesh.field.setNumber(1, "VOut", global_mesh_size)
    gmsh.model.mesh.field.setNumber(1, "XMin", 0.0)
    gmsh.model.mesh.field.setNumber(1, "XMax", Lc)
    gmsh.model.mesh.field.setNumber(1, "YMin", y0)
    gmsh.model.mesh.field.setNumber(1, "YMax", y1)
    gmsh.model.mesh.field.setNumber(1, "ZMin", -1.0)
    gmsh.model.mesh.field.setNumber(1, "ZMax", 1.0)
    gmsh.model.mesh.field.setNumber(1, "Thickness", channel_mesh_size)
    gmsh.model.mesh.field.setAsBackgroundMesh(1)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.model.mesh.generate(2)

    points, tri_cells, domain, surface_edges, surface_owners = extract_mesh_from_gmsh(
        volume_phys_ids, active_curve_phys
    )
    gmsh.finalize()
    return points, tri_cells, domain, surface_edges, surface_owners, surface_names


def build_full_square_tissue_mesh(
    L,
    Lc,
    y0,
    y1,
    tissue_domain_id,
    global_mesh_size,
    channel_mesh_size,
    tol,
    uniform_tissue=False,
):
    """
    Full square tissue mesh for MPC.

    uniform_tissue=False: refine near the channel strip (VIn=channel_mesh_size).
    uniform_tissue=True: use global_mesh_size everywhere (for background h-studies).
    """
    gmsh.initialize()
    gmsh.model.add("darcy_mpc_tissue_2d")
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)

    gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, L)
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(2, [1], tag=tissue_domain_id)
    gmsh.model.setPhysicalName(2, tissue_domain_id, "tissue")

    surface_names = ["left", "right", "bottom", "top"]
    boundary_curves = {name: [] for name in surface_names}
    for dim, tag in gmsh.model.getEntities(1):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(dim, tag)
        name = classify_curve(xmin, ymin, xmax, ymax, L, Lc, y0, y1, tol, explicit=False)
        if name in boundary_curves:
            boundary_curves[name].append(tag)

    curve_phys_ids = {"left": 11, "right": 12, "bottom": 13, "top": 14}
    active_curve_phys = {}
    for name in surface_names:
        tags = boundary_curves[name]
        if not tags:
            gmsh.finalize()
            raise RuntimeError(f"No boundary curves found for '{name}'")
        pid = curve_phys_ids[name]
        gmsh.model.addPhysicalGroup(1, tags, tag=pid)
        gmsh.model.setPhysicalName(1, pid, name)
        active_curve_phys[name] = pid

    if uniform_tissue:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", global_mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", global_mesh_size)
    else:
        gmsh.model.mesh.field.add("Box", 1)
        gmsh.model.mesh.field.setNumber(1, "VIn", channel_mesh_size)
        gmsh.model.mesh.field.setNumber(1, "VOut", global_mesh_size)
        gmsh.model.mesh.field.setNumber(1, "XMin", 0.0)
        gmsh.model.mesh.field.setNumber(1, "XMax", Lc)
        gmsh.model.mesh.field.setNumber(1, "YMin", y0)
        gmsh.model.mesh.field.setNumber(1, "YMax", y1)
        gmsh.model.mesh.field.setNumber(1, "ZMin", -1.0)
        gmsh.model.mesh.field.setNumber(1, "ZMax", 1.0)
        gmsh.model.mesh.field.setNumber(1, "Thickness", channel_mesh_size)
        gmsh.model.mesh.field.setAsBackgroundMesh(1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.model.mesh.generate(2)

    points, tri_cells, domain, surface_edges, surface_owners = extract_mesh_from_gmsh(
        {tissue_domain_id}, active_curve_phys
    )
    gmsh.finalize()
    return points, tri_cells, domain, surface_edges, surface_owners, surface_names


def build_channel_line(L, Lc, line_mesh_size, fiber_domain_id):
    """Horizontal channel centerline from (0, L/2) to (Lc, L/2)."""
    n_nodes = max(2, int(np.round(Lc / line_mesh_size)) + 1)
    x = np.linspace(0.0, Lc, n_nodes)
    y = np.full(n_nodes, 0.5 * L)
    points = np.column_stack([x, y, np.zeros(n_nodes)])
    lines = np.column_stack([np.arange(n_nodes - 1), np.arange(1, n_nodes)]).astype(np.int64)
    domain = np.full(lines.shape[0], fiber_domain_id, dtype=np.int32)
    coupling_ids = list(range(n_nodes))
    return points, lines, domain, coupling_ids


def write_channel_fiber_mesh(output_dir, points, lines, domain, coupling_ids):
    n_cells = lines.shape[0]
    cells = np.hstack([np.full((n_cells, 1), 2, dtype=np.int64), lines]).ravel()
    celltypes = np.full(n_cells, pv.CellType.LINE, dtype=np.uint8)
    fiber = pv.UnstructuredGrid(cells, celltypes, points)
    fiber.point_data["GlobalNodeID"] = np.arange(1, points.shape[0] + 1, dtype=np.int32)
    fiber.cell_data["GlobalElementID"] = np.arange(1, n_cells + 1, dtype=np.int32)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vtu_path = output_dir / "channel.vtu"
    dat_path = output_dir / "channel.dat"
    nodes_path = output_dir / "channel_nodes.txt"
    fiber.save(str(vtu_path))
    np.savetxt(dat_path, domain, fmt="%d")
    nodes_1based = np.asarray(coupling_ids, dtype=np.int32) + 1
    np.savetxt(nodes_path, nodes_1based.reshape(1, -1), fmt="%d")

    print(
        f"- Channel fiber .vtu: {vtu_path}, points={fiber.n_points}, cells={fiber.n_cells}, "
        f"GlobalNodeID={fiber.point_data['GlobalNodeID'].dtype}, "
        f"GlobalElementID={fiber.cell_data['GlobalElementID'].dtype}"
    )
    print(f"- Channel domains: {dat_path}, dtype={domain.dtype}, rows={domain.size}")
    print(f"- Channel MPC nodes: {nodes_path}, n={nodes_1based.size}")
    return vtu_path, dat_path, nodes_path


def generate_mpc_case(
    square_size=1.0,
    channel_width=0.01,
    channel_length_fraction=0.75,
    global_mesh_size=0.05,
    channel_mesh_size=0.005,
    output_dir=None,
    tissue_domain_id=1,
    channel_fiber_domain_id=2,
    keep_existing_channel=False,
    uniform_tissue=False,
):
    """
    Generate full-square tissue + 1D channel meshes for MPC.

    If keep_existing_channel is True and channel.vtu already exists, reuse that
    1D line (for background-mesh convergence studies).
    """
    if square_size <= 0.0:
        raise ValueError("square_size must be > 0")
    if channel_width <= 0.0 or channel_width >= square_size:
        raise ValueError("channel_width must satisfy 0 < channel_width < square_size")
    if not (0.0 < channel_length_fraction < 1.0):
        raise ValueError("channel_length_fraction must satisfy 0 < fraction < 1")
    if global_mesh_size <= 0.0 or channel_mesh_size <= 0.0:
        raise ValueError("mesh sizes must be > 0")

    L = float(square_size)
    w = float(channel_width)
    Lc = float(channel_length_fraction) * L
    y0 = 0.5 * (L - w)
    y1 = 0.5 * (L + w)
    tol = 1e-4 * max(L, 1.0)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "mpc_mesh"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"MPC geometry: L={L}, channel=[{0}, {Lc}] x [{y0}, {y1}], "
        f"background_h={global_mesh_size}, channel_h={channel_mesh_size}, "
        f"uniform_tissue={uniform_tissue}"
    )

    channel_vtu = output_dir / "channel.vtu"
    channel_nodes = output_dir / "channel_nodes.txt"
    if keep_existing_channel and channel_vtu.exists() and channel_nodes.exists():
        fiber = pv.read(str(channel_vtu))
        fiber_pts = np.asarray(fiber.points, dtype=float)
        fiber_lines = []
        for i in range(fiber.n_cells):
            cell = fiber.get_cell(i)
            fiber_lines.append([int(cell.point_ids[0]), int(cell.point_ids[1])])
        fiber_lines = np.asarray(fiber_lines, dtype=np.int64)
        fiber_domain = np.loadtxt(output_dir / "channel.dat", dtype=np.int32)
        coupling_ids = [int(i) for i in np.loadtxt(channel_nodes, dtype=int).ravel() - 1]
        print(f"Reusing existing channel: {fiber_pts.shape[0]} nodes, {fiber_lines.shape[0]} lines")
    else:
        fiber_pts, fiber_lines, fiber_domain, coupling_ids = build_channel_line(
            L, Lc, channel_mesh_size, channel_fiber_domain_id
        )
        write_channel_fiber_mesh(output_dir, fiber_pts, fiber_lines, fiber_domain, coupling_ids)

    points, tri_cells, domain, surface_edges, surface_owners, surface_names = build_full_square_tissue_mesh(
        L=L,
        Lc=Lc,
        y0=y0,
        y1=y1,
        tissue_domain_id=tissue_domain_id,
        global_mesh_size=global_mesh_size,
        channel_mesh_size=channel_mesh_size,
        tol=tol,
        uniform_tissue=uniform_tissue,
    )
    domain[:] = tissue_domain_id
    write_sv_mesh(output_dir, points, tri_cells, domain, surface_edges, surface_owners, surface_names)

    print(
        f"- Channel line: n_nodes={fiber_pts.shape[0]}, n_elements={fiber_lines.shape[0]}, "
        f"actual_h={Lc / max(fiber_pts.shape[0] - 1, 1):.6g}"
    )
    print("- Remaining issues: none")
    return {
        "n_tissue_points": int(points.shape[0]),
        "n_tissue_cells": int(tri_cells.shape[0]),
        "n_channel_points": int(fiber_pts.shape[0]),
        "n_channel_cells": int(fiber_lines.shape[0]),
        "global_mesh_size": float(global_mesh_size),
        "channel_mesh_size": float(channel_mesh_size),
    }


def generate_explicit_case(
    square_size=1.0,
    channel_width=0.01,
    channel_length_fraction=0.75,
    global_mesh_size=0.05,
    channel_mesh_size=0.005,
    output_dir=None,
    tissue_domain_id_in=1,
    channel_domain_id_in=2,
):
    """Generate the explicit (channel-cutout) tissue mesh at given sizes."""
    import sys

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "explicit_mesh"
    else:
        output_dir = Path(output_dir)

    if square_size <= 0.0:
        raise ValueError("square_size must be > 0")
    if channel_width <= 0.0 or channel_width >= square_size:
        raise ValueError("channel_width must satisfy 0 < channel_width < square_size")
    if not (0.0 < channel_length_fraction < 1.0):
        raise ValueError("channel_length_fraction must satisfy 0 < fraction < 1")
    if global_mesh_size <= 0.0 or channel_mesh_size <= 0.0:
        raise ValueError("mesh sizes must be > 0")

    # build_geometry_and_mesh reads these module globals.
    global L, w, Lc, y0, y1, tol, tissue_domain_id, channel_domain_id
    mod = sys.modules[__name__]
    mod.global_mesh_size = float(global_mesh_size)
    mod.channel_mesh_size = float(channel_mesh_size)

    L = float(square_size)
    w = float(channel_width)
    Lc = float(channel_length_fraction) * L
    y0 = 0.5 * (L - w)
    y1 = 0.5 * (L + w)
    tol = 1e-4 * max(L, 1.0)
    tissue_domain_id = int(tissue_domain_id_in)
    channel_domain_id = int(channel_domain_id_in)

    print(
        f"Explicit geometry: L={L}, channel=[{0}, {Lc}] x [{y0}, {y1}], "
        f"background_h={global_mesh_size}, channel_h={channel_mesh_size}"
    )

    points, tri_cells, domain, surface_edges, surface_owners, surface_names = build_geometry_and_mesh(
        include_channel_volume=False
    )
    write_sv_mesh(output_dir, points, tri_cells, domain, surface_edges, surface_owners, surface_names)
    return {
        "n_tissue_points": int(points.shape[0]),
        "n_tissue_cells": int(tri_cells.shape[0]),
        "global_mesh_size": float(global_mesh_size),
        "channel_mesh_size": float(channel_mesh_size),
    }


# ---------------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------------
square_size = 1.0
channel_width = 0.01
channel_length_fraction = 0.75
global_mesh_size = 0.05
channel_mesh_size = 0.005

base_dir = Path(__file__).resolve().parent
implicit_dir = base_dir / "mesh"
explicit_dir = base_dir / "explicit_mesh"
mpc_dir = base_dir / "mpc_mesh"

tissue_domain_id = 1
channel_domain_id = 2
channel_fiber_domain_id = 2

# Set DARCY_GEN_MESH_SKIP_MAIN=1 before importing this module from run_convergence.
import os

if os.environ.get("DARCY_GEN_MESH_SKIP_MAIN") != "1":
    if channel_width <= 0.0 or channel_width >= square_size:
        raise ValueError("channel_width must satisfy 0 < channel_width < square_size")
    if not (0.0 < channel_length_fraction < 1.0):
        raise ValueError("channel_length_fraction must satisfy 0 < fraction < 1")
    if global_mesh_size <= 0.0 or channel_mesh_size <= 0.0:
        raise ValueError("mesh sizes must be > 0")

    L = float(square_size)
    w = float(channel_width)
    Lc = float(channel_length_fraction) * L
    y0 = 0.5 * (L - w)
    y1 = 0.5 * (L + w)
    tol = 1e-4 * max(L, 1.0)

    print(
        f"Geometry: L={L}, channel=[{0}, {Lc}] x [{y0}, {y1}], "
        f"mesh sizes global={global_mesh_size}, channel={channel_mesh_size}"
    )

    points, tri_cells, domain, surface_edges, surface_owners, surface_names = build_geometry_and_mesh(
        include_channel_volume=True
    )
    write_sv_mesh(implicit_dir, points, tri_cells, domain, surface_edges, surface_owners, surface_names)

    generate_explicit_case(
        square_size=square_size,
        channel_width=channel_width,
        channel_length_fraction=channel_length_fraction,
        global_mesh_size=global_mesh_size,
        channel_mesh_size=channel_mesh_size,
        output_dir=explicit_dir,
    )

    generate_mpc_case(
        square_size=square_size,
        channel_width=channel_width,
        channel_length_fraction=channel_length_fraction,
        global_mesh_size=global_mesh_size,
        channel_mesh_size=channel_mesh_size,
        output_dir=mpc_dir,
        tissue_domain_id=tissue_domain_id,
        channel_fiber_domain_id=channel_fiber_domain_id,
        keep_existing_channel=False,
        uniform_tissue=False,
    )
