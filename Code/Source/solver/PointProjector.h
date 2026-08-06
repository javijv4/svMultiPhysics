// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef POINT_PROJECTOR_H
#define POINT_PROJECTOR_H

#include "Array.h"
#include "CmMod.h"
#include "ComMod.h"
#include "Simulation.h"
#include "Vector.h"
#include "utils.h"

#include <memory>
#include <string>
#include <vector>

class PointProjector {
  public:
    PointProjector(const std::string& name, int iM, int iFa, int jM, int jFa);
    virtual ~PointProjector() = default;

    virtual void setup(Simulation* simulation) = 0;
    virtual void distribute(ComMod& com_mod, CmMod& cm_mod, cmType& cm);
    virtual std::string coupling_method_name() const = 0;

    const std::string& name() const;

  protected:
    std::string projection_name_;
    int iM_ = -1;
    int iFa_ = -1;
    int jM_ = -1;
    int jFa_ = -1;
};

class EndNodeProjector : public PointProjector {
  public:
    EndNodeProjector(const std::string& name, int iM, int iFa, int jM, int jFa, double tol);

    void setup(Simulation* simulation) override;
    std::string coupling_method_name() const override;

    int num_target_nodes(const ComMod& com_mod) const;
    void merge_gN(ComMod& com_mod, std::vector<utils::stackType>& stk, utils::stackType& avNds);

  private:
    double tol_ = 0.0;
    utils::stackType lPrj_;
};

class MpcProjector : public PointProjector {
  public:
    struct ConstraintRow {
      int node1d = -1;
      std::vector<int> node3d;
      std::vector<double> weights;
    };

    MpcProjector(const std::string& name, int iM, int iFa, int jM, int jFa);

    void setup(Simulation* simulation) override;
    void distribute(ComMod& com_mod, CmMod& cm_mod, cmType& cm) override;
    std::string coupling_method_name() const override;

    std::vector<ConstraintRow> get_constraint_rows() const;
    bool has_constraints() const;

  private:
    bool project_to_mesh() const { return jFa_ < 0; }

    void match_points(const ComMod& com_mod, const mshType& src_mesh, const faceType& src_face,
                      const mshType& dst_mesh);

    Vector<int> target_element_;
    Vector<int> global_node1d_;
    Array<int> nodes3d_;
    Array<double> weights_;
    int gnNo_ = 0;
};

class PointProjectorManager {
  public:
    PointProjectorManager();
    ~PointProjectorManager();

    void create_from_parameters(Simulation* simulation);
    void setup_end_nodes(Simulation* simulation, utils::stackType& avNds);
    void setup_mpc(Simulation* simulation);
    void distribute(ComMod& com_mod, CmMod& cm_mod, cmType& cm);
    bool has_mpc() const;
    void apply_mpc_constraints(ComMod& com_mod, const Vector<int>& incL, const Vector<double>& res) const;

  private:
    std::vector<std::unique_ptr<PointProjector>> projectors_;
    std::vector<MpcProjector*> mpc_projectors_;
    std::vector<EndNodeProjector*> end_node_projectors_;
};

#endif
