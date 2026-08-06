#!/usr/bin/env python3
"""
Mesh convergence study: compare MPC vs EndNodes on successively refined tissue meshes.

Keeps the same 1D diagonal line and coupling nodes; regenerates only the tissue mesh.
"""

import os
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

# Import gen_mesh helpers without running its default generation.
os.environ["HEAT_GEN_MESH_SKIP_MAIN"] = "1"
from gen_mesh import generate_case  # noqa: E402


def interpolate_in_triangle(point, tri_points, tri_values):
    BARY_TOL = 1e-6
    v0 = tri_points[1] - tri_points[0]
    v1 = tri_points[2] - tri_points[0]
    v2 = point - tri_points[0]
    mat = np.column_stack([v0[:2], v1[:2]])
    try:
        c0, c1 = np.linalg.solve(mat, v2[:2])
        c2 = 1.0 - c0 - c1
        if all(-BARY_TOL <= c <= 1.0 + BARY_TOL for c in (c0, c1, c2)):
            return c2 * tri_values[0] + c0 * tri_values[1] + c1 * tri_values[2]
    except np.linalg.LinAlgError:
        pass
    return None


def triangle_connectivity(mesh):
    tris = []
    for i in range(mesh.n_cells):
        cell = mesh.get_cell(i)
        if cell.type == pv.CellType.TRIANGLE:
            tris.append(list(cell.point_ids))
    return np.asarray(tris, dtype=np.int64)


def mpc_constraint_error(out_dir, nodes_file, field_name="Temperature"):
    mpc_nodes = np.loadtxt(nodes_file, dtype=int).ravel() - 1
    result_files = sorted(Path(out_dir).glob("result_*.vtu"))
    if len(result_files) == 0:
        raise FileNotFoundError(f"No results in {out_dir}")
    timesteps = sorted(int(p.stem.split("_")[1]) for p in result_files)

    r0 = pv.read(str(result_files[0]))
    points = np.asarray(r0.points)
    mpc_pos = points[mpc_nodes]
    ien = triangle_connectivity(r0)
    field0 = np.asarray(r0.point_data[field_name])
    tri_ids = []
    for pos in mpc_pos:
        found = -1
        for e, tri in enumerate(ien):
            if interpolate_in_triangle(pos, points[tri], field0[tri]) is not None:
                found = e
                break
        if found < 0:
            raise RuntimeError("MPC node not located in tissue mesh")
        tri_ids.append(found)

    errs = []
    for ts in timesteps:
        r = pv.read(str(Path(out_dir) / f"result_{ts:03d}.vtu"))
        field = np.asarray(r.point_data[field_name])
        ien = triangle_connectivity(r)
        for i, pos in enumerate(mpc_pos):
            tri = ien[tri_ids[i]]
            v1 = field[mpc_nodes[i]]
            v2 = interpolate_in_triangle(pos, np.asarray(r.points)[tri], field[tri])
            errs.append(abs(v1 - v2))
    return float(np.max(errs)), float(np.mean(errs))


def probe_difference(out_mpc, out_end, nodes_file, coupling_index=1, field_name="Temperature"):
    mpc_nodes = np.loadtxt(nodes_file, dtype=int).ravel() - 1
    mpc_files = sorted(Path(out_mpc).glob("result_*.vtu"))
    end_files = sorted(Path(out_end).glob("result_*.vtu"))
    ts = sorted(
        set(int(p.stem.split("_")[1]) for p in mpc_files).intersection(
            int(p.stem.split("_")[1]) for p in end_files
        )
    )
    r0 = pv.read(str(Path(out_mpc) / f"result_{ts[0]:03d}.vtu"))
    fiber_pt = np.asarray(r0.points)[mpc_nodes[coupling_index]]
    order = np.argsort(np.linalg.norm(np.asarray(r0.points) - fiber_pt, axis=1))
    probe = int(order[0])
    if probe == mpc_nodes[coupling_index] and len(order) > 1:
        probe = int(order[1])
    probe_xyz = np.asarray(r0.points)[probe]

    diffs = []
    for t in ts:
        rm = pv.read(str(Path(out_mpc) / f"result_{t:03d}.vtu"))
        re = pv.read(str(Path(out_end) / f"result_{t:03d}.vtu"))
        vm = np.asarray(rm.point_data[field_name])[probe]
        ie = int(np.argmin(np.linalg.norm(np.asarray(re.points) - probe_xyz, axis=1)))
        ve = np.asarray(re.point_data[field_name])[ie]
        diffs.append(abs(vm - ve))
    return float(np.max(diffs)), float(np.mean(diffs)), probe


# ---------------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------------
square_side = 1.0
n_line_nodes = 21
mesh_sizes = [0.2, 0.1, 0.05]
svmultiphysics = (
    "/home/javiera/software/svMultiPhysics/dev_MPC_OO/"
    "build/svMultiPhysics-build/bin/svmultiphysics"
)
python_exe = "/home/javiera/miniconda3/envs/ep-env/bin/python"

base_dir = Path(__file__).resolve().parent
mesh_dir = base_dir / "mesh"
conv_dir = base_dir / "convergence"
solver_mpc = base_dir / "solver_mpc.xml"
solver_end = base_dir / "solver_endnodes.xml"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if not Path(svmultiphysics).is_file():
    raise FileNotFoundError(f"svmultiphysics binary not found: {svmultiphysics}")

conv_dir.mkdir(parents=True, exist_ok=True)

# Build the shared 1D line once at the coarsest setting, then reuse it.
if mesh_dir.exists():
    shutil.rmtree(mesh_dir)
generate_case(
    square_side=square_side,
    mesh_size=mesh_sizes[0],
    n_line_nodes=n_line_nodes,
    output_dir=mesh_dir,
    keep_existing_line=False,
)

rows = []
for h in mesh_sizes:
    tag = f"h_{h:g}".replace(".", "p")
    case_dir = conv_dir / tag
    case_mesh = case_dir / "mesh"
    case_dir.mkdir(parents=True, exist_ok=True)

    if case_mesh.exists():
        shutil.rmtree(case_mesh)
    # Copy fixed line mesh, then regenerate tissue only
    shutil.copytree(mesh_dir, case_mesh)
    info = generate_case(
        square_side=square_side,
        mesh_size=h,
        n_line_nodes=n_line_nodes,
        output_dir=case_mesh,
        keep_existing_line=True,
    )

    # Copy solvers with relative mesh paths (run inside case_dir)
    for src_name, dst_name in (
        ("solver_mpc.xml", "solver_mpc.xml"),
        ("solver_endnodes.xml", "solver_endnodes.xml"),
    ):
        shutil.copy2(base_dir / src_name, case_dir / dst_name)

    out_mpc = case_dir / "out_mpc"
    out_end = case_dir / "out_end"
    for d in (out_mpc, out_end):
        if d.exists():
            shutil.rmtree(d)

    print(f"\n=== Running MPC for mesh_size={h} ===")
    subprocess.check_call([svmultiphysics, "solver_mpc.xml"], cwd=str(case_dir))
    print(f"=== Running EndNodes for mesh_size={h} ===")
    subprocess.check_call([svmultiphysics, "solver_endnodes.xml"], cwd=str(case_dir))

    nodes_file = case_mesh / "line_nodes.txt"
    max_mpc_err, mean_mpc_err = mpc_constraint_error(out_mpc, nodes_file)
    max_diff, mean_diff, probe = probe_difference(out_mpc, out_end, nodes_file)

    row = {
        "mesh_size": h,
        "n_tissue_points": info["n_tissue_points"],
        "n_tissue_cells": info["n_tissue_cells"],
        "mpc_max_error": max_mpc_err,
        "mpc_mean_error": mean_mpc_err,
        "mpc_end_max_diff": max_diff,
        "mpc_end_mean_diff": mean_diff,
        "probe_node": probe,
    }
    rows.append(row)
    print(row)

# Restore default mesh/ at base case resolution (last / medium)
generate_case(
    square_side=square_side,
    mesh_size=0.1,
    n_line_nodes=n_line_nodes,
    output_dir=mesh_dir,
    keep_existing_line=True,
)

# Summary CSV
csv_path = conv_dir / "convergence_summary.csv"
with open(csv_path, "w") as f:
    keys = list(rows[0].keys())
    f.write(",".join(keys) + "\n")
    for row in rows:
        f.write(",".join(str(row[k]) for k in keys) + "\n")
print(f"Wrote {csv_path}")

hs = np.array([r["mesh_size"] for r in rows], dtype=float)
mpc_err = np.array([r["mpc_max_error"] for r in rows], dtype=float)
form_diff = np.array([r["mpc_end_max_diff"] for r in rows], dtype=float)

plt.figure(figsize=(6, 4))
plt.loglog(hs, mpc_err, "ko-", label="MPC constraint max error")
plt.loglog(hs, form_diff, "rs--", label="|MPC - EndNodes| max at probe")
plt.xlabel("mesh size h")
plt.ylabel("error")
plt.gca().invert_xaxis()
plt.grid(True, which="both", ls=":")
plt.legend(loc="best")
plt.tight_layout()
plot_path = conv_dir / "convergence.png"
plt.savefig(str(plot_path), dpi=300)
print(f"Wrote {plot_path}")
