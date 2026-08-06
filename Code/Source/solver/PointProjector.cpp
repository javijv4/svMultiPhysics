// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "PointProjector.h"

#include "all_fun.h"
#include "consts.h"
#include "lapack_defs.h"
#include "nn.h"
#include "read_msh.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

PointProjector::PointProjector(const std::string& name, int iM, int iFa, int jM, int jFa)
  : projection_name_(name), iM_(iM), iFa_(iFa), jM_(jM), jFa_(jFa)
{
}

void PointProjector::distribute(ComMod& com_mod, CmMod& cm_mod, cmType& cm)
{
}

const std::string& PointProjector::name() const
{
  return projection_name_;
}

EndNodeProjector::EndNodeProjector(const std::string& name, int iM, int iFa, int jM, int jFa, double tol)
  : PointProjector(name, iM, iFa, jM, jFa), tol_(tol)
{
}

void EndNodeProjector::setup(Simulation* simulation)
{
  auto& com_mod = simulation->get_com_mod();
  const auto& face1 = com_mod.msh[iM_].fa[iFa_];
  const auto& face2 = com_mod.msh[jM_].fa[jFa_];
  read_msh_ns::match_faces(com_mod, face1, face2, tol_, lPrj_);
}

std::string EndNodeProjector::coupling_method_name() const
{
  return "EndNodes";
}

int EndNodeProjector::num_target_nodes(const ComMod& com_mod) const
{
  return com_mod.msh[iM_].fa[iFa_].nNo;
}

void EndNodeProjector::merge_gN(ComMod& com_mod, std::vector<utils::stackType>& stk, utils::stackType& avNds)
{
  while (true) {
    int ia, ja, i, j, k;

    if (!utils::pull_stack(lPrj_, ja)) {
      break;
    }
    if (!utils::pull_stack(lPrj_, ia)) {
      break;
    }

    i = com_mod.msh[iM_].gN[ia];
    j = com_mod.msh[jM_].gN[ja];

    if (i == -1) {
      if (j == -1) {
        if (!utils::pull_stack(avNds, k)) {
          k = com_mod.gtnNo;
          com_mod.gtnNo = com_mod.gtnNo + 1;
        }
        com_mod.msh[iM_].gN[ia] = k;
        com_mod.msh[jM_].gN[ja] = k;
        utils::push_stack(stk[k], {iM_, ia, jM_, ja});
      } else {
        com_mod.msh[iM_].gN[ia] = j;
        utils::push_stack(stk[j], {iM_, ia});
      }
    } else {
      if (j == -1) {
        com_mod.msh[jM_].gN[ja] = i;
        utils::push_stack(stk[i], {jM_, ja});
      } else {
        if (i == j) {
          continue;
        }
        if (i > j) {
          k = i;
          i = j;
          j = k;
        }

        while (true) {
          int kM;
          if (!utils::pull_stack(stk[j], ja)) {
            break;
          }
          if (!utils::pull_stack(stk[j], kM)) {
            break;
          }
          com_mod.msh[kM].gN[ja] = i;
          utils::push_stack(stk[i], {kM, ja});
        }
        utils::push_stack(avNds, j);
      }
    }
  }
}

MpcProjector::MpcProjector(const std::string& name, int iM, int iFa, int jM, int jFa)
  : PointProjector(name, iM, iFa, jM, jFa)
{
}

void MpcProjector::setup(Simulation* simulation)
{
  auto& com_mod = simulation->get_com_mod();
  auto& mesh1 = com_mod.msh[iM_];
  const auto& face1 = mesh1.fa[iFa_];
  const auto& mesh2 = com_mod.msh[jM_];

  if (!mesh1.lFib) {
    throw std::runtime_error("MPC projection '" + projection_name_ + "' must be defined on a 1D fiber face.");
  }

  bool has_mpc_nodes_file = false;
  if (iM_ < static_cast<int>(simulation->parameters.mesh_parameters.size())) {
    auto mesh_param = simulation->parameters.mesh_parameters[iM_];
    for (auto face_param : mesh_param->face_parameters) {
      if (face_param->name() == projection_name_ &&
          face_param->mpc_nodes_file_path.defined() &&
          face_param->mpc_nodes_file_path() != "") {
        has_mpc_nodes_file = true;
        break;
      }
    }
  }

  if (!has_mpc_nodes_file) {
    throw std::runtime_error("MPC projection '" + projection_name_ + "' requires Mpc_nodes_file_path on its target face.");
  }

  const int dst_eNoN = project_to_mesh() ? mesh2.eNoN : mesh2.fa[jFa_].eNoN;

  target_element_ = Vector<int>(face1.nNo);
  target_element_ = -1;
  global_node1d_ = Vector<int>(face1.nNo);
  nodes3d_ = Array<int>(dst_eNoN, face1.nNo);
  nodes3d_ = -1;
  weights_ = Array<double>(dst_eNoN, face1.nNo);
  gnNo_ = face1.nNo;

  match_points(com_mod, mesh1, face1, mesh2);
}

void MpcProjector::distribute(ComMod& com_mod, CmMod& cm_mod, cmType& cm)
{
  cm.bcast(cm_mod, &gnNo_);
  int nrows = nodes3d_.nrows();
  cm.bcast(cm_mod, &nrows);

  if (cm.slv(cm_mod) && gnNo_ > 0) {
    global_node1d_.resize(gnNo_);
    target_element_.resize(gnNo_);
    nodes3d_.resize(nrows, gnNo_);
    weights_.resize(nrows, gnNo_);
  }

  if (gnNo_ > 0) {
    cm.bcast(cm_mod, global_node1d_);
    cm.bcast(cm_mod, target_element_);
    cm.bcast(cm_mod, nodes3d_);
    cm.bcast(cm_mod, weights_);
  }
}

std::string MpcProjector::coupling_method_name() const
{
  return "MPC";
}

std::vector<MpcProjector::ConstraintRow> MpcProjector::get_constraint_rows() const
{
  std::vector<ConstraintRow> rows;
  const int nrows = nodes3d_.nrows();

  for (int a = 0; a < gnNo_; a++) {
    ConstraintRow row;
    row.node1d = global_node1d_(a);

    for (int b = 0; b < nrows; b++) {
      const int n3_global = nodes3d_(b, a);
      if (n3_global < 0) {
        continue;
      }
      row.node3d.push_back(n3_global);
      row.weights.push_back(weights_(b, a));
    }

    if (row.node1d >= 0 && !row.node3d.empty()) {
      rows.push_back(row);
    }
  }

  return rows;
}

bool MpcProjector::has_constraints() const
{
  return gnNo_ > 0 && global_node1d_.size() > 0 && nodes3d_.size() > 0;
}

void MpcProjector::match_points(const ComMod& com_mod, const mshType& src_mesh, const faceType& src_face,
                                const mshType& dst_mesh)
{
  const int nsd = com_mod.nsd;
  const bool to_mesh = project_to_mesh();
  const faceType* dst_face = to_mesh ? nullptr : &dst_mesh.fa[jFa_];

  // setup_mpc runs before DISTRIBUTE, so volume connectivity lives in gIEN/gnEl.
  const bool use_gien = to_mesh && dst_mesh.gIEN.size() > 0;
  const auto dst_eType = to_mesh ? dst_mesh.eType : dst_face->eType;
  const int eNoN = to_mesh ? dst_mesh.eNoN : dst_face->eNoN;
  const int nEl = to_mesh ? (use_gien ? dst_mesh.gnEl : dst_mesh.nEl) : dst_face->nEl;
  const Array<int>& IENd = to_mesh ? (use_gien ? dst_mesh.gIEN : dst_mesh.IEN) : dst_face->IEN;
  const int nsd_param = consts::element_dimension.at(dst_eType);
  const auto& gX = com_mod.x;

  Array<double> xib(2, nsd_param);
  Array<double> Nb(2, eNoN);
  nn::get_nn_bnds(nsd_param, dst_eType, eNoN, xib, Nb);

  const std::string dest_label = to_mesh ? ("mesh '" + dst_mesh.name + "'")
                                         : ("face '" + dst_face->name + "'");

  for (int a = 0; a < src_face.nNo; a++) {
    global_node1d_(a) = src_face.gN(a);

    Vector<double> xp(nsd);
    for (int i = 0; i < nsd; i++) {
      xp(i) = src_face.x(i, a) * src_mesh.scF;
    }

    bool located = false;
    Array<double> xl(nsd, eNoN);
    Vector<double> xi(nsd);
    Vector<double> N(eNoN);
    Array<double> Nx(nsd_param, eNoN);

    for (int e = 0; e < nEl; e++) {
      for (int b = 0; b < eNoN; b++) {
        int global_node_id;
        if (to_mesh) {
          const int mesh_node = IENd(b, e);
          global_node_id = dst_mesh.gN[mesh_node];
        } else {
          global_node_id = IENd(b, e);
        }
        for (int i = 0; i < nsd; i++) {
          xl(i, b) = gX(i, global_node_id);
        }
      }

      bool ok = false;
      try {
        nn::get_xi(nsd_param, dst_eType, eNoN, xl, xp, xi, ok);
      } catch (const std::exception&) {
        continue;
      }

      bool inside = true;
      for (int i = 0; i < nsd_param; i++) {
        if (xi(i) < xib(0, i) || xi(i) > xib(1, i)) {
          inside = false;
          break;
        }
      }
      if (!inside) {
        continue;
      }

      nn::get_gnn(nsd_param, dst_eType, eNoN, xi, N, Nx);

      bool valid_shape = true;
      for (int b = 0; b < eNoN; b++) {
        if (N(b) < -1e-10) {
          valid_shape = false;
          break;
        }
      }
      if (!valid_shape) {
        continue;
      }

      target_element_(a) = e;

      if (!to_mesh) {
        int vol_elem = -1;
        if (dst_face->gE.size() > e && dst_face->gE(e) >= 0) {
          vol_elem = dst_face->gE(e);
        }

        bool use_vol_elem = (vol_elem >= 0 &&
            (dst_mesh.eType == consts::ElementType::TET4 || dst_mesh.eType == consts::ElementType::TET10));

        if (use_vol_elem) {
          int max_elem = (dst_mesh.gIEN.size() > 0) ? dst_mesh.gnEl : dst_mesh.nEl;
          if (vol_elem >= max_elem) {
            use_vol_elem = false;
          }
        }

        if (use_vol_elem) {
          const Array<int>* ien_ptr = nullptr;
          if (dst_mesh.gIEN.size() > 0 && vol_elem < dst_mesh.gnEl) {
            ien_ptr = &dst_mesh.gIEN;
          } else if (vol_elem < dst_mesh.nEl) {
            ien_ptr = &dst_mesh.IEN;
          } else {
            throw std::runtime_error("MPC: Volume element index out of bounds.");
          }

          const Array<int>& vol_ien = *ien_ptr;
          Vector<int> ptr(eNoN);
          std::vector<bool> setIt(dst_mesh.eNoN);
          std::fill(setIt.begin(), setIt.end(), true);

          for (int b = 0; b < eNoN; b++) {
            const int face_node_global = IENd(b, e);
            bool found = false;
            for (int ib = 0; ib < dst_mesh.eNoN; ib++) {
              if (setIt[ib]) {
                const int vol_node_mesh = vol_ien(ib, vol_elem);
                const int vol_node_global = dst_mesh.gN[vol_node_mesh];
                if (vol_node_global == face_node_global) {
                  ptr(b) = ib;
                  setIt[ib] = false;
                  found = true;
                  break;
                }
              }
            }
            if (!found) {
              throw std::runtime_error("MPC: Could not map face node to volume element node for tet element.");
            }
          }

          for (int b = 0; b < eNoN; b++) {
            const int vol_node_local = ptr(b);
            const int vol_node_mesh = vol_ien(vol_node_local, vol_elem);
            nodes3d_(b, a) = dst_mesh.gN[vol_node_mesh];
            weights_(b, a) = N(b);
          }
        } else {
          for (int b = 0; b < eNoN; b++) {
            nodes3d_(b, a) = IENd(b, e);
            weights_(b, a) = N(b);
          }
        }
      } else {
        for (int b = 0; b < eNoN; b++) {
          const int mesh_node = IENd(b, e);
          nodes3d_(b, a) = dst_mesh.gN[mesh_node];
          weights_(b, a) = N(b);
        }
      }

      located = true;
      break;
    }

    if (!located) {
      throw std::runtime_error("MPC node not found on projection " + dest_label +
                               " for projection '" + projection_name_ + "'.");
    }
  }
}

PointProjectorManager::PointProjectorManager()
{
}

PointProjectorManager::~PointProjectorManager() = default;

void PointProjectorManager::create_from_parameters(Simulation* simulation)
{
  auto& com_mod = simulation->get_com_mod();
  projectors_.clear();
  mpc_projectors_.clear();
  end_node_projectors_.clear();

  for (auto& params : simulation->parameters.projection_parameters) {
    const auto face1_name = params->name();

    int iM, iFa, jM, jFa;
    all_fun::find_face(com_mod.msh, face1_name, iM, iFa);

    const bool has_face = params->project_from_face.defined() && params->project_from_face() != "";
    const bool has_mesh = params->project_from_mesh.defined() && params->project_from_mesh() != "";

    if (has_face) {
      all_fun::find_face(com_mod.msh, params->project_from_face(), jM, jFa);
    } else if (has_mesh) {
      all_fun::find_msh(com_mod.msh, params->project_from_mesh(), jM);
      if (jM < 0) {
        throw std::runtime_error("Can't find mesh named '" + params->project_from_mesh() +
                                 "' for Add_projection '" + face1_name + "'.");
      }
      jFa = -1;
    } else {
      throw std::runtime_error("Add_projection '" + face1_name +
                               "' requires Project_from_face or Project_from_mesh.");
    }

    const auto method = params->coupling_method();
    if (method == "EndNodes") {
      if (jFa < 0) {
        throw std::runtime_error("Add_projection '" + face1_name +
                                 "' with Coupling_method EndNodes requires Project_from_face.");
      }
      auto projector = std::make_unique<EndNodeProjector>(face1_name, iM, iFa, jM, jFa, params->projection_tolerance());
      end_node_projectors_.push_back(projector.get());
      projectors_.push_back(std::move(projector));
    } else if (method == "MPC") {
      auto projector = std::make_unique<MpcProjector>(face1_name, iM, iFa, jM, jFa);
      mpc_projectors_.push_back(projector.get());
      projectors_.push_back(std::move(projector));
    } else {
      throw std::runtime_error("Unknown Coupling_method '" + method + "' for Add_projection '" + face1_name + "'.");
    }
  }
}

void PointProjectorManager::setup_end_nodes(Simulation* simulation, utils::stackType& avNds)
{
  auto& com_mod = simulation->get_com_mod();

  if (projectors_.empty()) {
    return;
  }

  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    auto& mesh = com_mod.msh[iM];
    mesh.gpN = Vector<int>(mesh.gnNo);
  }

  int nStk = 0;
  for (auto projector : end_node_projectors_) {
    nStk += projector->num_target_nodes(com_mod);
  }
  std::vector<utils::stackType> stk(nStk);

  for (auto projector : end_node_projectors_) {
    projector->setup(simulation);
    projector->merge_gN(com_mod, stk, avNds);
  }
}

void PointProjectorManager::setup_mpc(Simulation* simulation)
{
  for (auto projector : mpc_projectors_) {
    projector->setup(simulation);
  }
}

void PointProjectorManager::distribute(ComMod& com_mod, CmMod& cm_mod, cmType& cm)
{
  int n_mpc = static_cast<int>(mpc_projectors_.size());
  cm.bcast(cm_mod, &n_mpc);

  if (cm.slv(cm_mod)) {
    projectors_.clear();
    mpc_projectors_.clear();
    end_node_projectors_.clear();
    for (int i = 0; i < n_mpc; i++) {
      auto projector = std::make_unique<MpcProjector>("", -1, -1, -1, -1);
      mpc_projectors_.push_back(projector.get());
      projectors_.push_back(std::move(projector));
    }
  }

  for (auto projector : mpc_projectors_) {
    projector->distribute(com_mod, cm_mod, cm);
  }
}

bool PointProjectorManager::has_mpc() const
{
  for (auto projector : mpc_projectors_) {
    if (projector->has_constraints()) {
      return true;
    }
  }
  return false;
}

void PointProjectorManager::apply_mpc_constraints(ComMod& com_mod, const Vector<int>& incL,
                                                  const Vector<double>& res) const
{
  if (!has_mpc()) {
    return;
  }

  auto& eq = com_mod.eq[com_mod.cEq];
  if (!(eq.phys == consts::EquationType::phys_CEP || eq.phys == consts::EquationType::phys_heatS)) {
    return;
  }

  if (com_mod.dof < 1) {
    return;
  }

  constexpr int mpc_dof = 0;

  std::unordered_map<int, int> gtl_map;
  for (int local = 0; local < com_mod.lhs.mynNo; local++) {
    int global_node = com_mod.ltg(local);
    gtl_map[global_node] = local;
  }

  auto gtl = [&gtl_map](int global_node) -> int {
    auto it = gtl_map.find(global_node);
    return (it != gtl_map.end()) ? it->second : -1;
  };

  std::vector<MpcProjector::ConstraintRow> rows;
  for (auto projector : mpc_projectors_) {
    auto projector_rows = projector->get_constraint_rows();
    rows.insert(rows.end(), projector_rows.begin(), projector_rows.end());
  }

  const int m = static_cast<int>(rows.size());
  if (m == 0) {
    return;
  }

  auto eval_Bx_local = [&](Vector<double>& y_out) {
    y_out.resize(m);
    y_out = 0.0;
    for (int i = 0; i < m; i++) {
      double val = 0.0;
      const auto& row = rows[i];

      for (int k = 0; k < static_cast<int>(row.node3d.size()); k++) {
        const int n3_local = gtl(row.node3d[k]);
        if (n3_local >= 0) {
          val += row.weights[k] * com_mod.R(mpc_dof, n3_local);
        }
      }

      const int n1_local = gtl(row.node1d);
      if (n1_local >= 0) {
        val -= com_mod.R(mpc_dof, n1_local);
      }

      y_out(i) = val;
    }
  };

  // Use the same BC face activation (incL/res) as the subsequent unconstrained solve,
  // otherwise the Schur complement is built for a different operator than A.
  Array<double> R_orig = com_mod.R;
  Array<double> Val_orig = com_mod.Val;

  com_mod.R = R_orig;
  com_mod.Val = Val_orig;
  eq.linear_algebra->solve(com_mod, eq, incL, res);

  Vector<double> g_local;
  eval_Bx_local(g_local);
  CmMod cm_mod;
  Vector<double> g = com_mod.cm.reduce(cm_mod, g_local, MPI_SUM);

  Array<double> S(m, m);
  S = 0.0;

  for (int j = 0; j < m; j++) {
    com_mod.R = 0.0;
    com_mod.Val = Val_orig;
    const auto& rowj = rows[j];

    for (int k = 0; k < static_cast<int>(rowj.node3d.size()); k++) {
      const int n3_local = gtl(rowj.node3d[k]);
      if (n3_local >= 0) {
        com_mod.R(mpc_dof, n3_local) += rowj.weights[k];
      }
    }

    const int n1_local = gtl(rowj.node1d);
    if (n1_local >= 0) {
      com_mod.R(mpc_dof, n1_local) -= 1.0;
    }

    eq.linear_algebra->solve(com_mod, eq, incL, res);

    Vector<double> y_local;
    eval_Bx_local(y_local);
    Vector<double> y = com_mod.cm.reduce(cm_mod, y_local, MPI_SUM);

    for (int i = 0; i < m; i++) {
      S(i, j) = y(i);
    }
  }

  Vector<int> ipiv(m);
  int nrhs = 1;
  int info = 0;
  Array<double> rhs_lambda(m, 1);
  for (int i = 0; i < m; i++) {
    rhs_lambda(i, 0) = g(i);
  }

  dgesv_(&m, &nrhs, S.data(), &m, ipiv.data(), rhs_lambda.data(), &m, &info);
  if (info != 0) {
    com_mod.R = R_orig;
    com_mod.Val = Val_orig;
    throw std::runtime_error("[PointProjectorManager::apply_mpc_constraints] Failed to solve MPC Schur complement system (DGESV).");
  }

  com_mod.R = R_orig;
  com_mod.Val = Val_orig;
  for (int j = 0; j < m; j++) {
    const double lam = rhs_lambda(j, 0);
    const auto& rowj = rows[j];

    for (int k = 0; k < static_cast<int>(rowj.node3d.size()); k++) {
      const int n3_local = gtl(rowj.node3d[k]);
      if (n3_local >= 0) {
        com_mod.R(mpc_dof, n3_local) -= lam * rowj.weights[k];
      }
    }

    const int n1_local = gtl(rowj.node1d);
    if (n1_local >= 0) {
      com_mod.R(mpc_dof, n1_local) += lam;
    }
  }

  com_mod.Val = Val_orig;
}
