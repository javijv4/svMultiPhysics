// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "fiber_loader.h"

#include <gtest/gtest.h>

#include <vtkFloatArray.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkTetra.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>

#include <cmath>
#include <filesystem>
#include <stdexcept>

TEST(FiberLoader, InterpolatesPointDataAtQuadraturePoints)
{
  vtkNew<vtkPoints> points;
  points->InsertNextPoint(0.0, 0.0, 0.0);
  points->InsertNextPoint(1.0, 0.0, 0.0);
  points->InsertNextPoint(0.0, 1.0, 0.0);
  points->InsertNextPoint(0.0, 0.0, 1.0);

  vtkNew<vtkTetra> tetra;
  for (int i = 0; i < 4; ++i) {
    tetra->GetPointIds()->SetId(i, i);
  }

  vtkNew<vtkFloatArray> fibers;
  fibers->SetName("FIB_DIR");
  fibers->SetNumberOfComponents(3);
  fibers->InsertNextTuple3(1.0, 0.0, 0.0);
  fibers->InsertNextTuple3(1.0, 1.0, 0.0);
  fibers->InsertNextTuple3(1.0, 0.0, 1.0);
  fibers->InsertNextTuple3(1.0, 1.0, 1.0);

  vtkNew<vtkUnstructuredGrid> grid;
  grid->SetPoints(points);
  grid->InsertNextCell(tetra->GetCellType(), tetra->GetPointIds());
  grid->GetPointData()->AddArray(fibers);

  const auto file_name =
      std::filesystem::temp_directory_path() / "svmp_nodal_fiber_test.vtu";
  vtkNew<vtkXMLUnstructuredGridWriter> writer;
  writer->SetFileName(file_name.string().c_str());
  writer->SetInputData(grid);
  writer->SetDataModeToAscii();
  ASSERT_EQ(writer->Write(), 1);

  mshType mesh;
  mesh.name = "unit tetrahedron";
  mesh.gnEl = 1;
  mesh.gnNo = 4;
  mesh.eNoN = 4;
  mesh.nG = 1;
  mesh.nFn = 1;
  mesh.gIEN.resize(4, 1);
  mesh.N.resize(4, 1);
  mesh.fN.resize(3, 1);
  for (int a = 0; a < 4; ++a) {
    mesh.gIEN(a, 0) = a;
    mesh.N(a, 0) = 0.25;
  }

  EXPECT_TRUE(fiber_loader::load(file_name.string(), "FIB_DIR", 0, 3, mesh));
  std::filesystem::remove(file_name);

  const double norm = std::sqrt(1.5);
  EXPECT_NEAR(mesh.fN_q(0, 0, 0), 1.0 / norm, 1.0e-7);
  EXPECT_NEAR(mesh.fN_q(1, 0, 0), 0.5 / norm, 1.0e-7);
  EXPECT_NEAR(mesh.fN_q(2, 0, 0), 0.5 / norm, 1.0e-7);
}

TEST(FiberLoader, ReorthogonalizesAtEveryQuadraturePointAndPreservesFiber)
{
  mshType mesh;
  mesh.gnEl = 1;
  mesh.nG = 2;
  mesh.nFn = 2;
  mesh.fN.resize(6, 1);
  mesh.fN_q.resize(6, 2, 1);

  // At the first point, the sheet contains a component along the fiber.
  mesh.fN_q(0, 0, 0) = 2.0;
  mesh.fN_q(1, 0, 0) = 0.0;
  mesh.fN_q(2, 0, 0) = 0.0;
  mesh.fN_q(3, 0, 0) = 1.0;
  mesh.fN_q(4, 0, 0) = 1.0;
  mesh.fN_q(5, 0, 0) = 0.0;

  // Use a different, non-axis-aligned basis at the second point.
  mesh.fN_q(0, 1, 0) = 1.0;
  mesh.fN_q(1, 1, 0) = 1.0;
  mesh.fN_q(2, 1, 0) = 0.0;
  mesh.fN_q(3, 1, 0) = 0.0;
  mesh.fN_q(4, 1, 0) = 2.0;
  mesh.fN_q(5, 1, 0) = 1.0;

  fiber_loader::finalize_interpolated(3, mesh);

  for (int g = 0; g < mesh.nG; ++g) {
    double fiber_norm_squared = 0.0;
    double sheet_norm_squared = 0.0;
    double dot = 0.0;
    for (int i = 0; i < 3; ++i) {
      fiber_norm_squared += mesh.fN_q(i, g, 0) * mesh.fN_q(i, g, 0);
      sheet_norm_squared += mesh.fN_q(3 + i, g, 0) * mesh.fN_q(3 + i, g, 0);
      dot += mesh.fN_q(i, g, 0) * mesh.fN_q(3 + i, g, 0);
    }
    EXPECT_NEAR(fiber_norm_squared, 1.0, 1.0e-14);
    EXPECT_NEAR(sheet_norm_squared, 1.0, 1.0e-14);
    EXPECT_NEAR(dot, 0.0, 1.0e-14);
  }

  EXPECT_NEAR(mesh.fN_q(0, 1, 0), 1.0 / std::sqrt(2.0), 1.0e-14);
  EXPECT_NEAR(mesh.fN_q(1, 1, 0), 1.0 / std::sqrt(2.0), 1.0e-14);
  EXPECT_DOUBLE_EQ(mesh.fN_q(2, 1, 0), 0.0);

  // The element-level compatibility value mirrors the first quadrature point.
  for (int i = 0; i < 6; ++i) {
    EXPECT_DOUBLE_EQ(mesh.fN(i, 0), mesh.fN_q(i, 0, 0));
  }
}

TEST(FiberLoader, RejectsDegenerateInterpolatedDirection)
{
  mshType mesh;
  mesh.gnEl = 1;
  mesh.nG = 1;
  mesh.nFn = 1;
  mesh.fN.resize(3, 1);
  mesh.fN_q.resize(3, 1, 1);

  EXPECT_THROW(fiber_loader::finalize_interpolated(3, mesh),
               std::runtime_error);
}
