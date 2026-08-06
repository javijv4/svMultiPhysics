# Object-Oriented `PointProjector` design for `dev_MPC_OO`

## Context

`dev_MPC` implements 1D-fiber-to-3D multipoint-constraint (MPC) coupling for CEP, as an alternative to today's "glue matching nodes into one shared DOF" mechanism (`set_projector`, driven by `<Add_projection>` XML blocks). That branch has fallen behind `main`, and its MPC code is written as free functions bolted onto existing procedural code and structs (`ComMod.h`'s `faceType` gained a dozen MPC-specific fields, `read_msh.cpp` gained `set_projector_mpc`, `distribute.cpp` gained a ~100-line save/broadcast/restore dance, `eq_assem.cpp` gained a Schur-complement solve keyed off a global `mpcFlag`). Rather than rebasing that as-is, the goal is to design a proper object-oriented `PointProjector` abstraction in `dev_MPC_OO` (currently at `main`'s tip) that:

- Has exactly two interchangeable strategies for coupling a 1D fiber face to a 3D face: **EndNodes** (today's node-merge behavior) and **MPC** (ported from `dev_MPC`, re-architected).
- Consolidates the projection-setup logic that's currently spread across `read_msh.cpp`, `distribute.cpp`, and the linear-solve path into one class hierarchy + manager, instead of free functions and a global bool flag.
- Explicitly scopes out `set_ris_projector`/`Add_RIS_projection` and `set_uris_meshes`/`Add_URIS_*` (and unrelated point-search code in `remesh.cpp`/`contact.cpp`) — those stay untouched.
- Requires the coupling method to be spelled out explicitly per `<Add_projection>` block (no default, no inference from which face file path happens to be populated, unlike `dev_MPC`).

This document is the design and implementation checklist for the OO projector refactor.

## Current mechanism being replaced

`<Add_projection name="X"><Project_from_face>Y</Project_from_face><Projection_tolerance>tol</Projection_tolerance></Add_projection>` → parsed into `ProjectionParameters` (`Code/Source/solver/Parameters.h:891`) → consumed by `set_projector()` (`Code/Source/solver/read_msh.cpp:2127`), which resolves both faces by name (`all_fun::find_face`), calls `match_faces()` (block-binned nearest-neighbor node matcher, `read_msh.cpp:863`), then runs a union-find merge over `mesh.gN` (each mesh's local→global node-id array) using a shared free-list `avNds` threaded in from `read_msh()`. This is how e.g. a 1D Purkinje fiber mesh's terminal face gets DOF-shared with a 3D tissue face for CEP coupling (`tests/cases/cep/cylinder_purkinje_1d3d/solver.xml`).

`dev_MPC` adds a second, mutually-exclusive-by-inference mechanism: `Mpc_nodes_file_path` on `FaceParameters`, `set_projector_mpc()` + `match_point_face()` (point-in-face-element localization + interpolation weights, *not* a DOF merge — the 1D and 3D nodes keep independent global IDs), a Schur-complement hard-constraint solve in `eq_assem::modify_eq_assem_for_mpc()` called every solve iteration, and MPC bookkeeping fields bolted directly onto `faceType`/`ComMod`.

## Design decisions (confirmed with user)

1. **Scope**: only the `Add_projection`/`set_projector` family. RIS/URIS untouched.
2. **Only two strategies**: EndNodes and MPC, designed as polymorphic subclasses so a third could be added later without touching call sites.
3. **Explicit XML selection, no default**: a new required `<Coupling_method>` child element on `<Add_projection>`, value `EndNodes` or `MPC`. Missing or unrecognized → hard parse error. No inference from which file path is populated.
4. Naming: base class `PointProjector` (chosen for this narrower scope, to avoid confusion with RIS/URIS "projection").

## Class hierarchy

New files: `Code/Source/solver/PointProjector.h` / `.cpp` (added to `Code/Source/solver/CMakeLists.txt`, one line, next to the `ris.h/uris.h` pair). LAPACK is already linked by the solver, so the Schur-complement solve only needs the local `lapack_defs.h` declarations; do not add new linker dependencies for this refactor.

```cpp
class PointProjector {
  public:
    PointProjector(const std::string& name, int iM, int iFa, int jM, int jFa);
    virtual ~PointProjector() = default;
    virtual void setup(Simulation* simulation) = 0;
    virtual void distribute(ComMod& com_mod, CmMod& cm_mod, cmType& cm) {}   // default no-op
    virtual std::string coupling_method_name() const = 0;
    const std::string& name() const;
  protected:
    std::string projection_name_;
    int iM_, iFa_, jM_, jFa_;   // resolved target/source face indices
};

class EndNodeProjector : public PointProjector {
  public:
    EndNodeProjector(const std::string& name, int iM, int iFa, int jM, int jFa, double tol);
    void setup(Simulation* simulation) override;   // resolves faces, runs match_faces() into an internal lPrj
    std::string coupling_method_name() const override { return "EndNodes"; }
    // Not part of the polymorphic interface — see note below:
    void merge_gN(ComMod& com_mod, std::vector<utils::stackType>& stk, utils::stackType& avNds);
  private:
    double tol_;
};

class MpcProjector : public PointProjector {
  public:
    struct ConstraintRow { int node1d; std::vector<int> node3d; std::vector<double> weights; };
    MpcProjector(const std::string& name, int iM, int iFa, int jM, int jFa);
    void setup(Simulation* simulation) override;                              // was set_projector_mpc + match_point_face
    void distribute(ComMod& com_mod, CmMod& cm_mod, cmType& cm) override;      // was distribute.cpp's snapshot/broadcast
    std::string coupling_method_name() const override { return "MPC"; }
    std::vector<ConstraintRow> get_constraint_rows() const;
  private:
    Vector<int> target_element_, global_node1d_;
    Array<int> nodes3d_;
    Array<double> weights_;
    int gnNo_ = 0;
};

class PointProjectorManager {
  public:
    PointProjectorManager();
    ~PointProjectorManager();   // out-of-line; owns unique_ptr<PointProjector>
    void create_from_parameters(Simulation* simulation);          // parses Coupling_method, builds+dispatches subclasses
    void setup_end_nodes(Simulation* simulation, utils::stackType& avNds);   // replaces set_projector
    void setup_mpc(Simulation* simulation);                        // replaces set_projector_mpc; run after combined global face IDs exist
    void distribute(ComMod& com_mod, CmMod& cm_mod, cmType& cm);   // replaces the mpcFlag broadcast block
    bool has_mpc() const;                                          // replaces com_mod.mpcFlag
    void apply_mpc_constraints(ComMod& com_mod) const;             // replaces eq_assem::modify_eq_assem_for_mpc
  private:
    std::vector<std::unique_ptr<PointProjector>> projectors_;
    std::vector<MpcProjector*> mpc_projectors_;       // non-owning, for aggregation
    std::vector<EndNodeProjector*> end_node_projectors_;
};
```

**Design notes worth calling out explicitly (documented as comments in the header, not hidden):**

- **EndNodes merging is inherently cross-instance** (one shared `gN` numbering + one shared free-list `avNds` across *every* EndNodes projection block, exactly like today). So `EndNodeProjector::setup()` only does its own per-instance work (resolve faces, run `match_faces` into its own node-pair list); the actual union-find merge is a manager-orchestrated step (`setup_end_nodes` sizes one shared `stk` vector across all `EndNodeProjector`s and drives the merge, calling each instance's `merge_gN(...)` helper). This is the one place the design isn't "one virtual method does it all," and that's because the underlying algorithm is genuinely cross-object, not a modeling shortcut.
- **MPC is fully self-contained per instance** — no shared state at setup time. But the Schur-complement solve is inherently global (one assembled system `A` per equation), so if multiple `Add_projection` blocks use `Coupling_method: MPC` simultaneously, their constraint rows must be combined into **one** Schur solve, not one per object. `MpcProjector::get_constraint_rows()` returns only its own rows; `PointProjectorManager::apply_mpc_constraints()` is the sole place that concatenates rows from every `MpcProjector` and performs the single shared `dgesv_`-based solve (ported from `eq_assem::modify_eq_assem_for_mpc`).
- **MPC bookkeeping moves off `faceType`/`ComMod.h` entirely** and becomes private members of `MpcProjector`. This is a real simplification, not just tidiness: today (`dev_MPC`), `distribute.cpp` must snapshot each face's `mpc_*` arrays *before* `part_face` destroys/rebuilds the face during partitioning, broadcast, then restore them onto the rebuilt face (~100 lines). Since `MpcProjector` objects live independently of the `mshType`/`faceType` object graph, they are never touched by `part_face`, so `MpcProjector::distribute()` is just a direct broadcast of its own members (~10 lines) — no snapshot/restore needed at all.
- No `ComMod::mpcFlag`. `PointProjectorManager::has_mpc()` is computed from whether any `MpcProjector` actually produced constraint data, eliminating a class of "flag out of sync with state" bugs.

## Ownership

`ComMod` already has an out-of-line destructor (`~ComMod();` at `ComMod.h:1567`, defined in `ComMod.cpp`), so `ComMod.h` can forward-declare `class PointProjectorManager;` and hold `std::unique_ptr<PointProjectorManager> point_projectors;` without including `PointProjector.h` at all — constructed in `ComMod`'s constructor body in `ComMod.cpp` (which does include `PointProjector.h` fully). This avoids adding a new header dependency to `ComMod.h`, which is included nearly everywhere in the solver.

## Call-site integration

- **`read_msh.cpp`** (`read_msh()`): delete the `set_projector(simulation, avNds)` call; insert `com_mod.point_projectors->create_from_parameters(simulation);` then `->setup_end_nodes(simulation, avNds);` in its place, before the final `gN` assignment. The existing `gN` finalization loop (assigns fresh IDs to any node still `-1` after merging) stays exactly where it is, unchanged — both strategies, and meshes with no projection at all, still need it. Do **not** run MPC setup immediately after `gN` finalization. Run `com_mod.point_projectors->setup_mpc(simulation);` later, immediately after `com_mod.x` is rebuilt and all face `gN`/`IEN` arrays have been converted to combined global IDs; MPC mapping must see the same global IDs used by the assembled linear system. Delete `set_projector`, `set_projector_mpc`, `match_point_face` from `read_msh.cpp`/`.h` entirely (logic moves into `PointProjector.cpp`). **Keep `match_faces` in `read_msh.cpp`/`.h` untouched** — reused as-is by `EndNodeProjector`.
- **`load_msh.cpp`**: add `read_mpc_nodes()` (parallel free function to `read_ndnlff`, orthogonal to this refactor — just "how to populate `face.x`/`gN` from a node-id file"). In `read_sv()`, do **not** infer strategy from which file path is set (unlike `dev_MPC`): a fiber face just reads whichever of `End_nodes_face_file_path`/`Mpc_nodes_file_path` is populated using the matching reader; if a face is later referenced by an `Add_projection` block whose `Coupling_method` needs a file path that wasn't provided, that's caught as an explicit error in the relevant `PointProjector` subclass's `setup()`, not inferred here.
- **`distribute.cpp`**: delete the (not-yet-present-on-main, so really "don't add") snapshot/broadcast block; add one call `com_mod.point_projectors->distribute(com_mod, cm_mod, cm);` right after the existing `dist_uris(com_mod, cm_mod, cm);` call (`distribute.cpp:160`) — same position RIS/URIS distribution already uses, runs in both serial and parallel (harmless no-op broadcast in serial).
- **`Integrator.cpp`** (`Integrator::solve_linear_system()`): add the MPC hook immediately before `ls_ns::ls_solve(...)`: `com_mod.point_projectors->apply_mpc_constraints(com_mod)`. This replaces the stale pre-`Integrator` hook location and keeps the constraint projection adjacent to the linear solve it modifies.
- **`eq_assem.cpp`/`.h`**: no MPC code added here at all — the Schur-complement solve (RHS save/restore around `eq.linear_algebra->solve`, `B*x` evaluation via global-to-local node lookup, `dgesv_`, RHS projection) lives entirely in `PointProjectorManager::apply_mpc_constraints()` in `PointProjector.cpp` from the start. This keeps `eq_assem.cpp` scoped to actual residual/matrix assembly, not linear-algebra-level constraint handling.

## XML / Parameters changes

`ProjectionParameters` (`Code/Source/solver/Parameters.h:891`, `Parameters.cpp`) gains a required `Parameter<std::string> coupling_method` (XML tag `Coupling_method`), validated against a small allow-list (`"EndNodes"`, `"MPC"`) inside `set_values()`, raising `svmp::ParseException` on an unknown value. `set_values()` currently never calls the base class's `check_required()` (`Parameters.h:389`) — add that call so a missing `Coupling_method` (or the already-required-but-currently-unenforced `Project_from_face`) raises a clear error instead of silently defaulting. This is a small behavior change worth flagging: it starts enforcing `Project_from_face`'s pre-existing but dead `required=true`.

`FaceParameters` keeps both `end_nodes_face_file_path` (existing) and gains `mpc_nodes_file_path` (from `dev_MPC`, unchanged) — these describe *which file has this face's node list*, orthogonal to *which projector strategy consumes it*.

Cross-validation lives in `MpcProjector::setup()`: it must look up the target face's `FaceParameters` (walking `simulation->parameters.mesh_parameters[...]->face_parameters`, reusing `dev_MPC`'s `is_mpc_face_by_name`-style traversal) and throw a clear error if `Mpc_nodes_file_path` wasn't set, and a hard error (not a silent skip, unlike `dev_MPC`) if the face's mesh isn't `lFib` — since the strategy is now explicit, there's no ambiguity to silently tolerate.

## Existing test XML migration

9 files currently use `<Add_projection>` and have no `Coupling_method` today — all need `<Coupling_method> EndNodes </Coupling_method>` added to keep passing (verified via `grep -rl "Add_projection" tests/cases`):
`fsi/pipe_3d`, `fsi/pipe_3d_petsc`, `fsi/pipe_3d_trilinos_ml`, `fsi/pipe_3d_trilinos_bj`, `fsi/pipe_RCR_3d`, `fsi_ustruct/pipe_RCR_3d`, `fsi_ustruct/pipe_3d`, `uris/pipe_uris_fsi`, `cep/cylinder_purkinje_1d3d`. The last one is the only 1D-fiber case and a plausible future MPC candidate, but stays on `EndNodes` — don't repurpose it.

`dev_MPC` shipped MPC test fixtures that were never wired into pytest (`git show origin/dev_MPC:tests/test_cep.py | grep -i mpc` -> nothing): `tests/cases/cep/cylinder_purkinje_1d3d_mpc/` (full mesh + `solver.xml` + reference `result_001.vtu`), plus `tests/cases/cep/simple_slab/solver_mpc.xml` and `tests/cases/cep/slab_purkinje/solver_tet_mpc.xml`. Port `cylinder_purkinje_1d3d_mpc` into `dev_MPC_OO` (updating its `solver.xml` to add the now-required `<Coupling_method> MPC </Coupling_method>`) and add a `test_cylinder_purkinje_1d3d_mpc` entry to `tests/test_cep.py`, mirroring the existing `test_cylinder_purkinje_1d3d` pattern — this gives the MPC path its first real automated regression coverage.

Also port `dev_MPC/tests/cases/cep/slab_purkinje` as a required development/acceptance check. Bring over the mesh, MPC/end-node solver XMLs, and checker scripts, but omit generated `out_*` directories, logs, images, restart files, and other run products. Update XML/checkers for current `main` output naming, using `Membrane_potential` instead of old-branch `Action_potential` where needed. Convert the slab checker from print-only statistics to pass/fail and assert the maximum interpolation constraint error is `<= 1e-8`.

## Ordered implementation steps

1. `Parameters.h`/`.cpp`: add `ProjectionParameters::coupling_method` + allow-list validation + `check_required()` call; add `FaceParameters::mpc_nodes_file_path`.
2. `CMakeLists.txt`: add `PointProjector.h PointProjector.cpp`.
3. Write `PointProjector.h`: `PointProjector`, `EndNodeProjector`, `MpcProjector`, `PointProjectorManager`.
4. Write `PointProjector.cpp`: port `match_faces`-consuming union-find merge into `EndNodeProjector`/`setup_end_nodes`; port `set_projector_mpc`/`match_point_face` into `MpcProjector::setup`; port the distribute broadcast into `MpcProjector::distribute`; port the Schur solve into `apply_mpc_constraints`; implement `create_from_parameters` (the single place that dispatches `Coupling_method` strings to concrete subclasses).
5. `ComMod.h`/`.cpp`: forward-declare `PointProjectorManager`, add the `unique_ptr` member, construct it in `ComMod`'s constructor.
6. `read_msh.cpp`/`.h`: remove `set_projector`/`set_projector_mpc`/`match_point_face`; wire in the manager calls; keep `match_faces`.
7. `load_msh.cpp`: add `read_mpc_nodes`; drop strategy inference from `read_sv()`.
8. `distribute.cpp`: add the single `point_projectors->distribute(...)` call.
9. `Integrator.cpp`: call `point_projectors->apply_mpc_constraints(...)` inside `Integrator::solve_linear_system()`, immediately before `ls_ns::ls_solve(...)`.
10. Update the 9 existing test XML files; port `cylinder_purkinje_1d3d_mpc` fixture + add its pytest entry.
11. Port `dev_MPC/tests/cases/cep/slab_purkinje` as a required development check, including mesh, MPC/end-node XMLs, and pass/fail checker scripts, while excluding generated result directories and run artifacts.
12. Build; run the full test suite touching `Add_projection` (`pytest -k "pipe_3d or pipe_RCR_3d or pipe_uris_fsi or cylinder_purkinje_1d3d"`) to confirm EndNodes behavior is unchanged, plus the new cylinder MPC test.
13. Run `slab_purkinje` serial MPC and checker, then run the parallel MPC case, preferably `-np 3`, to exercise distribution.
14. Add negative-path smoke checks: missing `Coupling_method`, unknown value, `MPC` on a face without `Mpc_nodes_file_path`, `MPC` on a non-fiber mesh — all should fail with clear errors.

### Critical files
- `Code/Source/solver/PointProjector.h` / `.cpp` (new)
- `Code/Source/solver/read_msh.cpp` / `.h`
- `Code/Source/solver/ComMod.h` / `.cpp`
- `Code/Source/solver/Parameters.h` / `.cpp`
- `Code/Source/solver/load_msh.cpp`
- `Code/Source/solver/distribute.cpp`
- `Code/Source/solver/eq_assem.cpp` / `.h`
- `Code/Source/solver/Integrator.cpp`
- `Code/Source/solver/CMakeLists.txt`
- `tests/test_cep.py`, `tests/cases/cep/cylinder_purkinje_1d3d_mpc/*`, `tests/cases/cep/slab_purkinje/*`, the 9 existing `Add_projection` test XMLs
