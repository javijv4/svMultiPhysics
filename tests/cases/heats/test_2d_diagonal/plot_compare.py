#!/usr/bin/env python3
"""
Compare Temperature evolution at a tissue node near a coupling point for MPC vs EndNodes.
"""

from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


# ---------------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------------
out_mpc = "out_mpc"
out_end = "out_end"
field_name = "Temperature"
mpc_nodes_file = "mesh/line_nodes.txt"
coupling_index = 1
plot_file = "compare_mpc_endnodes.png"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
base = Path(__file__).resolve().parent
mpc_nodes = np.loadtxt(base / mpc_nodes_file, dtype=int).ravel() - 1

mpc_files = sorted(glob(str(base / out_mpc / "result_*.vtu")))
end_files = sorted(glob(str(base / out_end / "result_*.vtu")))
if len(mpc_files) == 0:
    raise FileNotFoundError(f"No results in {base / out_mpc}")
if len(end_files) == 0:
    raise FileNotFoundError(f"No results in {base / out_end}")

ts_mpc = sorted(int(Path(f).stem.split("_")[1]) for f in mpc_files)
ts_end = sorted(int(Path(f).stem.split("_")[1]) for f in end_files)
timesteps = sorted(set(ts_mpc).intersection(ts_end))
if len(timesteps) == 0:
    raise RuntimeError("No common timesteps between MPC and EndNodes outputs")

res0 = pv.read(str(base / out_mpc / f"result_{timesteps[0]:03d}.vtu"))
fiber_point = np.asarray(res0.points)[mpc_nodes[coupling_index]]

dists = np.linalg.norm(np.asarray(res0.points) - fiber_point, axis=1)
order = np.argsort(dists)
probe_node = int(order[0])
if probe_node == mpc_nodes[coupling_index] and len(order) > 1:
    probe_node = int(order[1])

print(f"Coupling fiber node (0-based): {mpc_nodes[coupling_index]} at {fiber_point}")
print(f"Probe tissue node: {probe_node} at {res0.points[probe_node]}")
print(f"Distance: {dists[probe_node]:.3e}")

vals_mpc = []
vals_end = []
for ts in timesteps:
    r_mpc = pv.read(str(base / out_mpc / f"result_{ts:03d}.vtu"))
    r_end = pv.read(str(base / out_end / f"result_{ts:03d}.vtu"))
    vals_mpc.append(np.asarray(r_mpc.point_data[field_name])[probe_node])
    d_end = np.linalg.norm(np.asarray(r_end.points) - np.asarray(res0.points)[probe_node], axis=1)
    end_node = int(np.argmin(d_end))
    vals_end.append(np.asarray(r_end.point_data[field_name])[end_node])

vals_mpc = np.asarray(vals_mpc)
vals_end = np.asarray(vals_end)
diff = np.abs(vals_mpc - vals_end)
print(f"Max |MPC - EndNodes| at probe: {diff.max():.6e}")
print(f"Mean |MPC - EndNodes| at probe: {diff.mean():.6e}")

np.savetxt(base / "compare_probe_diff.txt", [diff.max(), diff.mean()])

plt.figure(figsize=(5, 3.5))
plt.plot(timesteps, vals_mpc, "k-", lw=2, label="MPC")
plt.plot(timesteps, vals_end, "r--", lw=2, label="EndNodes")
plt.xlabel("Timestep")
plt.ylabel(field_name)
plt.title(f"Tissue node {probe_node} near coupling {coupling_index}")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(str(base / plot_file), dpi=300)
print(f"Wrote {plot_file}")
