// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef FIBER_LOADER_H
#define FIBER_LOADER_H

#include "ComMod.h"

#include <string>

namespace fiber_loader {

/// Load one fiber family from cell or point data in a VTU file.
///
/// Cell data is copied to every quadrature point. Point data is interpolated
/// with the element shape functions. Returns true when point data was used.
bool load(const std::string& file_name, const std::string& data_name, int idx,
          int nsd, mshType& mesh);

/// Normalize and reorthogonalize interpolated fiber bases at every quadrature
/// point. The primary fiber direction is normalized but otherwise preserved.
void finalize_interpolated(int nsd, mshType& mesh);

}  // namespace fiber_loader

#endif
