import meshio as io
import numpy as np
from pathlib import Path
from glob import glob

# Interpolate 3D values at mpc_node positions using tetrahedral linear basis
def interpolate_in_tet(point, tet_points, tet_values):
    """
    Interpolate a value at a point using barycentric coordinates in a tetrahedron.
    Returns interpolated value or None if point not in tetrahedron.
    """
    BARY_TOL = 1e-6

    # Vectors from first vertex to others
    v0 = tet_points[1] - tet_points[0]
    v1 = tet_points[2] - tet_points[0]
    v2 = tet_points[3] - tet_points[0]
    v3 = point - tet_points[0]
    
    # Solve for barycentric coordinates: v3 = c0*v0 + c1*v1 + c2*v2
    mat = np.column_stack([v0, v1, v2])
    try:
        coords = np.linalg.solve(mat, v3)
        c0, c1, c2 = coords
        c3 = 1.0 - c0 - c1 - c2
        
        # Check if point is inside tetrahedron
        if all(-BARY_TOL <= c <= 1 + BARY_TOL for c in [c0, c1, c2, c3]):
            # Linear interpolation using barycentric coordinates
            value = c3 * tet_values[0] + c0 * tet_values[1] + c1 * tet_values[2] + c2 * tet_values[3]
            return value
    except np.linalg.LinAlgError:
        pass
    
    return None


# Load mpc_nodes
out_fldr = 'out_serial'
field_name = "Membrane_potential"
max_allowed_error = 1e-8
mpc_nodes = np.loadtxt('mesh/purkinje_endnodes.txt', dtype=int)
mpc_nodes = mpc_nodes - 1  # Convert to zero-based indexing
n_mpc_nodes = len(mpc_nodes)

# Get list of all timesteps
result_files = sorted(glob(f'{out_fldr}/result_*.vtu'))
timesteps = sorted([int(Path(f).stem.split('_')[1]) for f in result_files])
timesteps = np.arange(10, 201, 10)  # Assuming timesteps from 10 to 1000 with step 10
n_timesteps = len(timesteps)

print(f"Found {n_timesteps} timesteps: {timesteps[:5]}...{timesteps[-5:]}")
print(f"Found {n_mpc_nodes} MPC nodes")

# Create arrays to store results
values_1d = np.zeros((n_mpc_nodes, n_timesteps))
values_3d_interp = np.zeros((n_mpc_nodes, n_timesteps))

# Get mpc_node positions
results = io.read(f'{out_fldr}/result_{timesteps[0]:03d}.vtu')
points = results.points
mpc_positions = points[mpc_nodes]
ap_values = results.point_data[field_name]
    

# Interpolate 3D values at mpc_node positions
ien_slab = results.cells_dict["tetra"]
tet_ids = np.zeros(n_mpc_nodes, dtype=int)
for i, mpc_pos in enumerate(mpc_positions):
    value = None
    # Search through all tetrahedra
    for e, tet_id in enumerate(ien_slab):
        tet_points = points[tet_id]
        tet_values = ap_values[tet_id]
        value = interpolate_in_tet(mpc_pos, tet_points, tet_values)
        if value is not None:
            tet_ids[i] = e
            print(f"Node {i} found in tet {e+76} at initial timestep")
            break

# Evaluate at each timestep
for ts_idx, ts in enumerate(timesteps):
    print(f"Processing timestep {ts} ({ts_idx+1}/{n_timesteps})...", end=' ')
    
    results = io.read(f'{out_fldr}/result_{ts:03d}.vtu')
    ien_slab = results.cells_dict["tetra"]
    
    # Get mesh data
    ap_values = results.point_data[field_name]
    
    
    # Store 1D values
    values_1d[:, ts_idx] = ap_values[mpc_nodes]
    
    # Interpolate 3D values at mpc_node positions
    for i, mpc_pos in enumerate(mpc_positions):
        value = None
        tet_id = tet_ids[i]
        tet_points = points[ien_slab[tet_id]]
        tet_values = ap_values[ien_slab[tet_id]]
        value = interpolate_in_tet(mpc_pos, tet_points, tet_values)
        values_3d_interp[i, ts_idx] = value if value is not None else np.nan
    
    print("done")

# Compute errors
errors = np.abs(values_1d - values_3d_interp)

# Print summary
print(f"\n{'='*70}")
print(f"MPC Constraint Evaluation Summary")
print(f"{'='*70}")
print(f"Timesteps: {timesteps[0]} to {timesteps[-1]}")
print(f"MPC nodes: {n_mpc_nodes}")
print(f"\nError Statistics (across all timesteps and nodes):")
max_error = np.nanmax(errors)
print(f"  Max error:    {max_error:.6e}")
print(f"  Mean error:   {np.nanmean(errors):.6e}")
print(f"  Std error:    {np.nanstd(errors):.6e}")
print(f"  Min error:    {np.nanmin(errors):.6e}")

print(f"\nPer-node statistics:")
print(f"{'Node ID':<8} {'Mean Error':<15} {'Max Error':<15} {'Std Error':<15}")
print("-" * 53)
for i in range(n_mpc_nodes):
    mean_err = np.nanmean(errors[i, :])
    max_err = np.nanmax(errors[i, :])
    std_err = np.nanstd(errors[i, :])
    print(f"{mpc_nodes[i]:<8} {mean_err:<15.6e} {max_err:<15.6e} {std_err:<15.6e}")

#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(4,3))
plt.plot(timesteps, values_1d.T, color='k', label='1D', lw=3)
plt.plot(timesteps, values_3d_interp.T, color='r', ls='--', label='3D Interpolated')
plt.xlabel('Timestep')
plt.ylabel('Temperature')
plt.legend(loc='best', handles=plt.gca().get_lines()[::4])
plt.savefig('mpc_evaluation_temperature.png', dpi=300)
# plt.show()
