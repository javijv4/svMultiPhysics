// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef FIBERS_H
#define FIBERS_H

#include "Array.h"
#include "Array3.h"

#include "eigen3/Eigen/Core"

class ComMod;
class CmMod;
class MeshParameters;
class cmType;
class mshType;

/**
 * @brief Fiber directions associated with one mesh.
 */
class Fibers {
  public:

    /// Location of the fiber directions in the input.
    enum class DataLocation {
      none,
      /// The fiber directions are associated with the mesh nodes.
      point,
      /// The fiber directions are associated with the mesh cells.
      cell,
      /// The fiber directions are constant vectors defined in the input file.
      constant
    };

    /**
     * @brief Returns the location of the fiber directions in the input.
     */
    DataLocation get_data_location() const { return data_location; }

    /// Number of fiber directions provided as input.
    int nfibers = 0;

    /**
     * @brief Reads fiber directions from file paths or constant vectors.
     */
    void read(const MeshParameters& mesh_parameters, mshType& mesh, int nsd);

    /**
     * @brief Parallel distribution of fiber directions.
     */
    void distribute(ComMod& com_mod, const CmMod& cm_mod, const cmType& cm,
      mshType& mesh);

    /**
     * @brief Clears the fiber directions.
     */
    void clear();

    /**
     * @brief Returns an Array view of the complete nsd x nsd orthonormal fiber
     * basis for a quadrature point and element.
     *
     * Columns are fiber, sheet/transverse, and (in 3D) sheet normal. Constant
     * directions use the single stored basis and cell directions use the
     * requested cell basis. Point directions are interpolated on the fly from
     * the element nodal fibers and mesh shape functions at quadrature index g;
     * that view aliases an internal scratch buffer and is invalidated by the
     * next point-fiber query. Constant and cell views share this object's
     * persistent storage and must not outlive it.
     */
    Array<double> get_quadrature_fibers(int nsd, int g, int e) const;

    /**
     * @brief Returns the complete nsd x nsd orthonormal fiber basis evaluated
     * with the supplied basis-function values.
     *
     * Point-associated directions are interpolated for element e; cell and
     * constant directions do not depend on the basis-function values and return
     * their stored basis. For point data the returned Array owns its storage.
     */
    Array<double> get_quadrature_fibers(int nsd,
      const Vector<double>& basis_functions, int e) const;

    /**
     * @brief Returns the complete input fiber bases.
     */
    const Array<double>& get_input_fibers() const { return input_fibers; }

    /**
     * @brief Gathers the complete input fiber bases in the original mesh order.
     *
     * Point bases are returned on the master process in point order. Cell bases
     * are gathered and reordered on the master process. Constant bases are
     * expanded to one basis per cell before being gathered.
     */
    Array<double> gather_fibers(const ComMod& com_mod,
      const CmMod& cm_mod, const mshType& mesh, int nsd) const;


    /**
     * @brief Returns a read-only Eigen view of the complete basis returned by
     * get_quadrature_fibers().
     *
     * The Map aliases this object's storage (persistent input data for
     * constant/cell, or the point-fiber scratch buffer) and must not outlive
     * it. For point data the view is invalidated by the next point-fiber query.
     */
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
    get_eigen_fibers(int nsd, int g, int e) const;

    /**
     * @brief Returns true if there are fiber directions.
     */
    bool has_fibers() const { return nfibers != 0; }

  protected:
    /// Fiber directions read from the input. Each column stores an nsd x nsd
    /// basis (nsd*nsd components). Column count depends on the data location:
    /// one for constant directions, one per cell for cell data, and one per
    /// point for point data.
    Array<double> input_fibers;

    /// Point-associated input fibers gathered at each element's nodes. Keeping
    /// this element-local representation allows on-the-fly interpolation with
    /// the mesh shape functions or alternative basis functions after the mesh
    /// has been partitioned.
    Array3<double> element_nodal_fibers;

    /// Mesh shape functions (eNoN x nG), used to interpolate point fibers at
    /// quadrature index g in get_quadrature_fibers(nsd, g, e).
    Array<double> shape_functions_;

    /// Scratch storage for the last on-the-fly point-fiber basis returned by
    /// get_quadrature_fibers(nsd, g, e) / get_eigen_fibers.
    mutable Array<double> quadrature_scratch_;

  private:
    DataLocation data_location = DataLocation::none;

    /**
     * @brief Reads fiber directions from VTK files. expected_location must be
     * point or cell and selects which FIB_DIR association is required.
     */
    void read_from_file(const MeshParameters& mesh_parameters, mshType& mesh,
        int nsd, DataLocation expected_location);

    /**
     * @brief Reads constant fiber directions as specified in the input file.
     */
    void read_directions(const MeshParameters& mesh_parameters, int nsd);

    /**
     * @brief Checks that the input fiber directions form an orthonormal set at
     * every stored location.
     *
     * Raises if any direction is not unit length or if any pair is not
     * orthogonal.
     */
    void check_input_fibers_orthonormal(int nsd) const;

    /**
     * @brief Builds the element-nodal point-fiber representation and stores the
     * mesh shape functions used for later on-the-fly interpolation.
     */
    void populate_element_nodal_fibers(const mshType& mesh, int nsd);

    /**
     * @brief Completes already-orthonormal input fibers to an nsd-dimensional
     * basis via cross product (3D) or 90-degree rotation (2D).
     */
    void complete_fiber_basis(Array<double>& fibers, int nsd);

};

#endif
