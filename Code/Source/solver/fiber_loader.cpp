// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "fiber_loader.h"

#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void validate_components(const vtkDataArray* data, const std::string& file_name,
                         const std::string& data_name, const int nsd)
{
  if (data->GetNumberOfComponents() < nsd) {
    throw std::runtime_error(
        "Fiber data '" + data_name + "' in '" + file_name + "' has " +
        std::to_string(data->GetNumberOfComponents()) +
        " components; at least " + std::to_string(nsd) + " are required.");
  }
}

double direction_norm(const int nsd, const int offset, const int g, const int e,
                      const mshType& mesh)
{
  double norm_squared = 0.0;
  for (int i = 0; i < nsd; ++i) {
    const double value = mesh.fN_q(offset + i, g, e);
    norm_squared += value * value;
  }
  return std::sqrt(norm_squared);
}

void normalize_direction(const int nsd, const int family, const int g,
                         const int e, mshType& mesh)
{
  const int offset = family * nsd;
  const double norm = direction_norm(nsd, offset, g, e, mesh);
  const double tolerance = std::sqrt(std::numeric_limits<double>::epsilon());
  if (norm <= tolerance) {
    throw std::runtime_error(
        "Interpolated fiber family " + std::to_string(family + 1) +
        " is undefined at element " + std::to_string(e + 1) +
        ", quadrature point " + std::to_string(g + 1) + ".");
  }
  for (int i = 0; i < nsd; ++i) {
    mesh.fN_q(offset + i, g, e) /= norm;
  }
}

}  // namespace

namespace fiber_loader {

bool load(const std::string& file_name, const std::string& data_name,
          const int idx, const int nsd, mshType& mesh)
{
  const int num_rows = mesh.nFn * nsd;
  if (idx < 0 || idx >= mesh.nFn) {
    throw std::runtime_error("Fiber family index " + std::to_string(idx) +
                             " is outside [0, " +
                             std::to_string(mesh.nFn) + ").");
  }
  if (mesh.fN.size() == 0) {
    mesh.fN.resize(num_rows, mesh.gnEl);
  }
  if (mesh.fN_q.size() == 0) {
    mesh.fN_q.resize(num_rows, mesh.nG, mesh.gnEl);
  }
  if (mesh.fN.nrows() != num_rows || mesh.fN.ncols() != mesh.gnEl ||
      mesh.fN_q.nrows() != num_rows || mesh.fN_q.ncols() != mesh.nG ||
      mesh.fN_q.nslices() != mesh.gnEl) {
    throw std::runtime_error(
        "Fiber storage is not sized for " + std::to_string(mesh.nFn) +
        " families, " + std::to_string(mesh.nG) +
        " quadrature points, and " + std::to_string(mesh.gnEl) +
        " elements.");
  }

  if (FILE* file = fopen(file_name.c_str(), "r")) {
    fclose(file);
  } else {
    throw std::runtime_error("The fiber direction VTK file '" + file_name +
                             "' can't be read.");
  }

  auto reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
  reader->SetFileName(file_name.c_str());
  reader->Update();
  vtkUnstructuredGrid* grid = reader->GetOutput();

  const vtkIdType num_nodes = grid->GetNumberOfPoints();
  if (num_nodes == 0) {
    throw std::runtime_error("Failed reading the VTK file '" + file_name + "'.");
  }

  const vtkIdType num_elems = grid->GetNumberOfCells();
  if (mesh.gnEl != num_elems) {
    throw std::runtime_error(
        "The number of elements (" + std::to_string(num_elems) +
        ") in the fiber direction VTK file '" + file_name +
        "' is not equal to the number of elements (" +
        std::to_string(mesh.gnEl) + ") for the mesh named '" + mesh.name +
        "'.");
  }

  vtkDataArray* fiber_data = grid->GetCellData()->GetArray(data_name.c_str());
  const bool point_data = fiber_data == nullptr;
  if (point_data) {
    fiber_data = grid->GetPointData()->GetArray(data_name.c_str());
  }
  if (fiber_data == nullptr) {
    throw std::runtime_error("No cell or point data named '" + data_name +
                             "' found in the fiber direction VTK file '" +
                             file_name + "'.");
  }
  validate_components(fiber_data, file_name, data_name, nsd);

  const int offset = idx * nsd;
  if (!point_data) {
    if (fiber_data->GetNumberOfTuples() != mesh.gnEl) {
      throw std::runtime_error(
          "Cell fiber data '" + data_name + "' in '" + file_name + "' has " +
          std::to_string(fiber_data->GetNumberOfTuples()) +
          " tuples; expected " + std::to_string(mesh.gnEl) + ".");
    }
    for (int e = 0; e < mesh.gnEl; ++e) {
      for (int i = 0; i < nsd; ++i) {
        const double value = fiber_data->GetComponent(e, i);
        mesh.fN(offset + i, e) = value;
        for (int g = 0; g < mesh.nG; ++g) {
          mesh.fN_q(offset + i, g, e) = value;
        }
      }
    }
    return false;
  }

  if (num_nodes != mesh.gnNo || fiber_data->GetNumberOfTuples() != mesh.gnNo) {
    throw std::runtime_error(
        "Point fiber data '" + data_name + "' in '" + file_name + "' has " +
        std::to_string(fiber_data->GetNumberOfTuples()) +
        " tuples on a grid with " + std::to_string(num_nodes) +
        " points; expected " + std::to_string(mesh.gnNo) +
        " points for mesh '" + mesh.name + "'.");
  }

  for (int e = 0; e < mesh.gnEl; ++e) {
    for (int g = 0; g < mesh.nG; ++g) {
      for (int i = 0; i < nsd; ++i) {
        double value = 0.0;
        for (int a = 0; a < mesh.eNoN; ++a) {
          const int node = mesh.gIEN(a, e);
          if (node < 0 || node >= mesh.gnNo) {
            throw std::runtime_error(
                "Mesh connectivity references invalid node " +
                std::to_string(node) + " in element " +
                std::to_string(e + 1) + ".");
          }
          value += mesh.N(a, g) * fiber_data->GetComponent(node, i);
        }
        mesh.fN_q(offset + i, g, e) = value;
      }
      normalize_direction(nsd, idx, g, e, mesh);
    }

    // Retain an element-level value for physics that have not yet adopted the
    // quadrature-point representation. Structural mechanics uses fN_q.
    for (int i = 0; i < nsd; ++i) {
      mesh.fN(offset + i, e) = mesh.fN_q(offset + i, 0, e);
    }
  }
  return true;
}

void finalize_interpolated(const int nsd, mshType& mesh)
{
  if (mesh.nFn > nsd) {
    throw std::runtime_error(
        "Cannot construct " + std::to_string(mesh.nFn) +
        " orthogonal fiber directions in " + std::to_string(nsd) +
        " spatial dimensions.");
  }

  for (int e = 0; e < mesh.gnEl; ++e) {
    for (int g = 0; g < mesh.nG; ++g) {
      // Modified Gram-Schmidt preserves the primary fiber direction: it is
      // only normalized, while every subsequent direction is adjusted.
      normalize_direction(nsd, 0, g, e, mesh);
      for (int family = 1; family < mesh.nFn; ++family) {
        const int offset = family * nsd;
        for (int previous = 0; previous < family; ++previous) {
          const int previous_offset = previous * nsd;
          double projection = 0.0;
          for (int i = 0; i < nsd; ++i) {
            projection += mesh.fN_q(offset + i, g, e) *
                          mesh.fN_q(previous_offset + i, g, e);
          }
          for (int i = 0; i < nsd; ++i) {
            mesh.fN_q(offset + i, g, e) -=
                projection * mesh.fN_q(previous_offset + i, g, e);
          }
        }
        normalize_direction(nsd, family, g, e, mesh);
      }
    }

    for (int family = 0; family < mesh.nFn; ++family) {
      for (int i = 0; i < nsd; ++i) {
        mesh.fN(family * nsd + i, e) =
            mesh.fN_q(family * nsd + i, 0, e);
      }
    }
  }
}

}  // namespace fiber_loader
