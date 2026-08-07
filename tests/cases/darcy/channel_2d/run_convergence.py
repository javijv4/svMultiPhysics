#!/usr/bin/env python3
"""
Background-mesh convergence study: MPC vs explicit Darcy channel problem.

At each background size h:
  - regenerate the explicit (channel-cutout) mesh
  - regenerate the MPC full-square tissue mesh (fixed 1D channel reused)
  - run both solvers
  - compare tissue pressure outside the channel region
"""

import os
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

os.environ["DARCY_GEN_MESH_SKIP_MAIN"] = "1"
from gen_mesh import generate_explicit_case, generate_mpc_case  # noqa: E402


def interpolate_in_triangle(point, tri_points, tri_values):
    BARY_TOL = 1e-6
    xy = np.asarray(point, dtype=float).ravel()[:2]
    tri_xy = np.asarray(tri_points, dtype=float)[:, :2]
    v0 = tri_xy[1] - tri_xy[0]
    v1 = tri_xy[2] - tri_xy[0]
    v2 = xy - tri_xy[0]
    mat = np.column_stack([v0, v1])
    try:
        c0, c1 = np.linalg.solve(mat, v2)
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


def split_fiber_tissue_point_ids(result):
    if "Mesh_ID" not in result.cell_data:
        return np.array([], dtype=np.int64), np.arange(result.n_points, dtype=np.int64)
    mesh_id = np.asarray(result.cell_data["Mesh_ID"])
    fiber_ids = set()
    tissue_ids = set()
    for ci in range(result.n_cells):
        pids = result.get_cell(ci).point_ids
        if int(mesh_id[ci]) == 0:
            fiber_ids.update(int(p) for p in pids)
        else:
            tissue_ids.update(int(p) for p in pids)
    return np.asarray(sorted(fiber_ids), dtype=np.int64), np.asarray(sorted(tissue_ids), dtype=np.int64)


def final_result(out_dir):
    files = sorted(Path(out_dir).glob("result_*.vtu"))
    if len(files) == 0:
        raise FileNotFoundError(f"No results in {out_dir}")
    return pv.read(str(files[-1]))


def sample_field_at_points(result, sample_xy, field_name="MBF", tissue_only=True):
    """Interpolate field at sample_xy using tissue triangles (fallback: nearest node)."""
    if tissue_only:
        _, tissue_ids = split_fiber_tissue_point_ids(result)
    else:
        tissue_ids = np.arange(result.n_points, dtype=np.int64)

    points = np.asarray(result.points)
    field = np.asarray(result.point_data[field_name]).ravel()
    ien = triangle_connectivity(result)
    # Restrict triangles to those fully in tissue point set when Mesh_ID exists.
    tissue_set = set(int(i) for i in tissue_ids)
    tissue_tris = [tri for tri in ien if all(int(n) in tissue_set for n in tri)]
    if len(tissue_tris) == 0:
        tissue_tris = list(ien)

    vals = np.full(len(sample_xy), np.nan, dtype=float)
    tissue_xy = points[tissue_ids, :2]
    tissue_field = field[tissue_ids]
    for i, xy in enumerate(sample_xy):
        pt = np.asarray(xy, dtype=float)
        found = None
        for tri in tissue_tris:
            found = interpolate_in_triangle(pt, points[tri], field[tri])
            if found is not None:
                break
        if found is None:
            j = int(np.argmin(np.linalg.norm(tissue_xy - pt, axis=1)))
            found = float(tissue_field[j])
        vals[i] = float(found)
    return vals


def outside_channel_mask(xy, Lc, y0, y1, margin=1e-8):
    x = xy[:, 0]
    y = xy[:, 1]
    in_channel = (x >= -margin) & (x <= Lc + margin) & (y >= y0 - margin) & (y <= y1 + margin)
    return ~in_channel


def compare_mpc_to_explicit(out_mpc, out_exp, Lc, y0, y1, field_name="MBF", max_samples=400):
    """
    Compare final tissue pressure: sample exterior explicit nodes, interpolate in MPC.
    Returns max/mean absolute and relative errors.
    """
    r_exp = final_result(out_exp)
    r_mpc = final_result(out_mpc)

    exp_pts = np.asarray(r_exp.points)[:, :2]
    exp_field = np.asarray(r_exp.point_data[field_name]).ravel()
    keep = outside_channel_mask(exp_pts, Lc, y0, y1)
    sample_xy = exp_pts[keep]
    sample_val = exp_field[keep]

    if sample_xy.shape[0] > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(sample_xy.shape[0], size=max_samples, replace=False)
        sample_xy = sample_xy[idx]
        sample_val = sample_val[idx]

    mpc_val = sample_field_at_points(r_mpc, sample_xy, field_name=field_name, tissue_only=True)
    abs_err = np.abs(mpc_val - sample_val)
    denom = np.maximum(np.abs(sample_val), 1e-12)
    rel_err = abs_err / denom
    return {
        "n_samples": int(sample_xy.shape[0]),
        "max_abs": float(np.max(abs_err)),
        "mean_abs": float(np.mean(abs_err)),
        "max_rel": float(np.max(rel_err)),
        "mean_rel": float(np.mean(rel_err)),
        "sample_xy": sample_xy,
        "exp_val": sample_val,
        "mpc_val": mpc_val,
    }


def write_solver_explicit(src_xml, dst_xml, out_folder="out_explicit"):
    text = Path(src_xml).read_text()
    if "<Save_results_in_folder>" not in text:
        text = text.replace(
            "<Convert_BIN_to_VTK_format> 1 </Convert_BIN_to_VTK_format>",
            "<Convert_BIN_to_VTK_format> 1 </Convert_BIN_to_VTK_format>\n"
            f"  <Save_results_in_folder> {out_folder} </Save_results_in_folder>",
        )
    else:
        # unlikely
        pass
    Path(dst_xml).write_text(text)


# ---------------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------------
square_size = 1.0
channel_width = 0.01
channel_length_fraction = 0.75
channel_mesh_size = 0.005
mesh_sizes = [0.1, 0.05, 0.025]

# Probes outside the channel strip for reporting.
probe_xy = [
    (0.25, 0.60),
    (0.50, 0.60),
    (0.70, 0.60),
    (0.50, 0.80),
    (0.90, 0.50),
]

svmultiphysics = (
    "/home/javiera/software/svMultiPhysics/dev_MPC_OO/"
    "build/svMultiPhysics-build/bin/svmultiphysics"
)

base_dir = Path(__file__).resolve().parent
shared_mpc = base_dir / "mpc_mesh"
conv_dir = base_dir / "convergence"
solver_mpc_src = base_dir / "darcy_sim_mpc.xml"
solver_exp_src = base_dir / "darcy_sim_explicit.xml"

L = float(square_size)
Lc = float(channel_length_fraction) * L
y0 = 0.5 * (L - float(channel_width))
y1 = 0.5 * (L + float(channel_width))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if not Path(svmultiphysics).is_file():
    raise FileNotFoundError(f"svmultiphysics binary not found: {svmultiphysics}")

conv_dir.mkdir(parents=True, exist_ok=True)

# Shared fixed 1D channel for all MPC cases.
if shared_mpc.exists():
    shutil.rmtree(shared_mpc)
generate_mpc_case(
    square_size=square_size,
    channel_width=channel_width,
    channel_length_fraction=channel_length_fraction,
    global_mesh_size=mesh_sizes[0],
    channel_mesh_size=channel_mesh_size,
    output_dir=shared_mpc,
    keep_existing_channel=False,
    uniform_tissue=True,
)

rows = []
for h in mesh_sizes:
    tag = f"h_{h:g}".replace(".", "p")
    case_dir = conv_dir / tag
    case_mpc = case_dir / "mpc_mesh"
    case_exp = case_dir / "explicit_mesh"
    case_dir.mkdir(parents=True, exist_ok=True)

    if case_mpc.exists():
        shutil.rmtree(case_mpc)
    if case_exp.exists():
        shutil.rmtree(case_exp)

    shutil.copytree(shared_mpc, case_mpc)
    mpc_info = generate_mpc_case(
        square_size=square_size,
        channel_width=channel_width,
        channel_length_fraction=channel_length_fraction,
        global_mesh_size=h,
        channel_mesh_size=channel_mesh_size,
        output_dir=case_mpc,
        keep_existing_channel=True,
        uniform_tissue=True,
    )
    exp_info = generate_explicit_case(
        square_size=square_size,
        channel_width=channel_width,
        channel_length_fraction=channel_length_fraction,
        global_mesh_size=h,
        channel_mesh_size=channel_mesh_size,
        output_dir=case_exp,
    )

    shutil.copy2(solver_mpc_src, case_dir / "darcy_sim_mpc.xml")
    write_solver_explicit(solver_exp_src, case_dir / "darcy_sim_explicit.xml")
    shutil.copy2(base_dir / "channel_pressure.dat", case_dir / "channel_pressure.dat")

    out_mpc = case_dir / "out_mpc"
    out_exp = case_dir / "out_explicit"
    for d in (out_mpc, out_exp):
        if d.exists():
            shutil.rmtree(d)

    print(f"\n=== Explicit for background h={h} ===")
    subprocess.check_call([svmultiphysics, "darcy_sim_explicit.xml"], cwd=str(case_dir))
    print(f"=== MPC for background h={h} ===")
    subprocess.check_call([svmultiphysics, "darcy_sim_mpc.xml"], cwd=str(case_dir))

    cmp = compare_mpc_to_explicit(out_mpc, out_exp, Lc=Lc, y0=y0, y1=y1)
    p_mpc = sample_field_at_points(final_result(out_mpc), probe_xy, tissue_only=True)
    p_exp = sample_field_at_points(final_result(out_exp), probe_xy, tissue_only=False)
    probe_abs = np.abs(p_mpc - p_exp)

    row = {
        "mesh_size": h,
        "n_mpc_tissue_points": mpc_info["n_tissue_points"],
        "n_mpc_tissue_cells": mpc_info["n_tissue_cells"],
        "n_exp_tissue_points": exp_info["n_tissue_points"],
        "n_exp_tissue_cells": exp_info["n_tissue_cells"],
        "n_channel_points": mpc_info["n_channel_points"],
        "mpc_vs_exp_max_abs": cmp["max_abs"],
        "mpc_vs_exp_mean_abs": cmp["mean_abs"],
        "mpc_vs_exp_max_rel": cmp["max_rel"],
        "mpc_vs_exp_mean_rel": cmp["mean_rel"],
        "probe_max_abs": float(np.max(probe_abs)),
        "probe_mean_abs": float(np.mean(probe_abs)),
        "n_compare_samples": cmp["n_samples"],
    }
    for i in range(len(probe_xy)):
        row[f"probe_{i}_mpc"] = float(p_mpc[i])
        row[f"probe_{i}_exp"] = float(p_exp[i])
    rows.append(row)
    print(row)

# Restore default meshes at project root.
generate_explicit_case(
    square_size=square_size,
    channel_width=channel_width,
    channel_length_fraction=channel_length_fraction,
    global_mesh_size=0.05,
    channel_mesh_size=channel_mesh_size,
    output_dir=base_dir / "explicit_mesh",
)
generate_mpc_case(
    square_size=square_size,
    channel_width=channel_width,
    channel_length_fraction=channel_length_fraction,
    global_mesh_size=0.05,
    channel_mesh_size=channel_mesh_size,
    output_dir=shared_mpc,
    keep_existing_channel=True,
    uniform_tissue=False,
)

csv_path = conv_dir / "convergence_summary.csv"
keys = list(rows[0].keys())
with open(csv_path, "w") as f:
    f.write(",".join(keys) + "\n")
    for row in rows:
        f.write(",".join(str(row[k]) for k in keys) + "\n")
print(f"Wrote {csv_path}")

hs = np.array([r["mesh_size"] for r in rows], dtype=float)
max_abs = np.array([r["mpc_vs_exp_max_abs"] for r in rows], dtype=float)
mean_abs = np.array([r["mpc_vs_exp_mean_abs"] for r in rows], dtype=float)
probe_max = np.array([r["probe_max_abs"] for r in rows], dtype=float)

plt.figure(figsize=(6.2, 4.2))
plt.loglog(hs, max_abs, "ko-", label="max |P_MPC - P_explicit|")
plt.loglog(hs, mean_abs, "bs--", label="mean |P_MPC - P_explicit|")
plt.loglog(hs, probe_max, "r^:", label="max |ΔP| at fixed probes")
plt.xlabel("background mesh size h")
plt.ylabel("error vs explicit")
plt.gca().invert_xaxis()
plt.grid(True, which="both", ls=":")
plt.legend(loc="best")
plt.tight_layout()
plot_path = conv_dir / "convergence.png"
plt.savefig(str(plot_path), dpi=300)
print(f"Wrote {plot_path}")
