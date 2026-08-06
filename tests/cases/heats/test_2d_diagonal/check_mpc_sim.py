#!/usr/bin/env python3
"""
Check MPC hard constraint: 1D coupling-node Temperature vs triangle-interpolated tissue values.
"""

from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


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
    if len(tris) == 0:
        raise RuntimeError("Expected triangle cells in result VTU")
    return np.asarray(tris, dtype=np.int64)


# ---------------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------------
out_fldr = "out_mpc"
field_name = "Temperature"
max_allowed_error = 1e-8
mpc_nodes_file = "mesh/line_nodes.txt"
plot_file = "mpc_evaluation.png"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
base = Path(__file__).resolve().parent
mpc_nodes = np.loadtxt(base / mpc_nodes_file, dtype=int).ravel() - 1
n_mpc_nodes = len(mpc_nodes)

result_files = sorted(glob(str(base / out_fldr / "result_*.vtu")))
if len(result_files) == 0:
    raise FileNotFoundError(f"No result_*.vtu files found in {base / out_fldr}")

timesteps = sorted(int(Path(f).stem.split("_")[1]) for f in result_files)
n_timesteps = len(timesteps)
print(f"Found {n_timesteps} timesteps: {timesteps[0]} ... {timesteps[-1]}")
print(f"Found {n_mpc_nodes} MPC nodes")

values_1d = np.zeros((n_mpc_nodes, n_timesteps))
values_2d_interp = np.zeros((n_mpc_nodes, n_timesteps))

results0 = pv.read(str(base / out_fldr / f"result_{timesteps[0]:03d}.vtu"))
points = np.asarray(results0.points)
mpc_positions = points[mpc_nodes]
field0 = np.asarray(results0.point_data[field_name])
ien = triangle_connectivity(results0)

tri_ids = np.full(n_mpc_nodes, -1, dtype=int)
for i, mpc_pos in enumerate(mpc_positions):
    for e, tri in enumerate(ien):
        value = interpolate_in_triangle(mpc_pos, points[tri], field0[tri])
        if value is not None:
            tri_ids[i] = e
            print(f"Node {i} (1D id {mpc_nodes[i]}) found in triangle {e}")
            break
    if tri_ids[i] < 0:
        raise RuntimeError(f"Could not locate MPC node {i} in tissue mesh")

for ts_idx, ts in enumerate(timesteps):
    results = pv.read(str(base / out_fldr / f"result_{ts:03d}.vtu"))
    field = np.asarray(results.point_data[field_name])
    ien = triangle_connectivity(results)
    values_1d[:, ts_idx] = field[mpc_nodes]
    for i, mpc_pos in enumerate(mpc_positions):
        tri = ien[tri_ids[i]]
        values_2d_interp[i, ts_idx] = interpolate_in_triangle(
            mpc_pos, np.asarray(results.points)[tri], field[tri]
        )

errors = np.abs(values_1d - values_2d_interp)
max_error = float(np.nanmax(errors))

print("=" * 70)
print("MPC Constraint Evaluation Summary")
print("=" * 70)
print(f"Timesteps: {timesteps[0]} to {timesteps[-1]}")
print(f"MPC nodes: {n_mpc_nodes}")
print(f"  Max error:    {max_error:.6e}")
print(f"  Mean error:   {np.nanmean(errors):.6e}")
print(f"  Std error:    {np.nanstd(errors):.6e}")
print(f"  Min error:    {np.nanmin(errors):.6e}")
print(f"\nPer-node statistics:")
print(f"{'Node ID':<8} {'Mean Error':<15} {'Max Error':<15} {'Std Error':<15}")
print("-" * 53)
for i in range(n_mpc_nodes):
    print(
        f"{mpc_nodes[i]:<8} {np.nanmean(errors[i, :]):<15.6e} "
        f"{np.nanmax(errors[i, :]):<15.6e} {np.nanstd(errors[i, :]):<15.6e}"
    )

if max_error > max_allowed_error:
    print(f"WARNING: max error {max_error:.3e} exceeds tolerance {max_allowed_error:.3e}")
else:
    print(f"OK: max error {max_error:.3e} <= {max_allowed_error:.3e}")

np.savetxt(base / out_fldr / "mpc_max_error.txt", [max_error])

plt.figure(figsize=(4, 3))
plt.plot(timesteps, values_1d.T, color="k", label="1D", lw=2)
plt.plot(timesteps, values_2d_interp.T, color="r", ls="--", label="2D Interpolated")
plt.xlabel("Timestep")
plt.ylabel(field_name)
handles = plt.gca().get_lines()
plt.legend(loc="best", handles=handles[:: max(1, n_mpc_nodes)])
plt.tight_layout()
plt.savefig(str(base / plot_file), dpi=300)
print(f"Wrote {plot_file}")
