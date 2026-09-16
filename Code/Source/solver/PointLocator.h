// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef POINT_LOCATOR_H
#define POINT_LOCATOR_H

#include "Array.h"
#include "Vector.h"

#include <optional>

namespace svmp {

/**
 * @brief Result of a nearest-point query.
 */
struct NearestPointResult {
  /// Index of the matched point in the locator point cloud, or -1 if unset.
  int point_index = -1;

  /// Euclidean distance from the query to the matched point.
  double distance = 0.0;
};

/**
 * @brief Spatial-query helper for a point cloud.
 *
 * The locator owns a copy of the supplied point coordinates. The current
 * implementation uses a deterministic linear scan; a spatial index can be
 * introduced later without changing the public query interface.
 */
class PointLocator {
 public:
  /**
   * @brief Construct a locator from a point cloud.
   *
   * @param[in] points Point coordinates, with one column per point.
   */
  explicit PointLocator(const Array<double>& points);

  /**
   * @brief Find the nearest stored point to a query.
   *
   * Distance ties keep the first visited candidate.
   *
   * @param[in] query Query coordinates. Its size must match the point-cloud
   *   dimension.
   * @param[in] exclude_index Optional point-cloud index to skip. Useful for
   *   same-mesh projections that must not map a node onto itself.
   * @return The nearest point, or nullopt if every candidate was excluded.
   */
  std::optional<NearestPointResult> find_nearest_neighbor(
      const Vector<double>& query,
      std::optional<int> exclude_index = std::nullopt) const;

 private:
  /// Point coordinates, with one column per point.
  Array<double> points_;

  /// Return the spatial dimension of the point cloud.
  int dimension() const { return points_.nrows(); }

  /// Return the number of points in the point cloud.
  int number_of_points() const { return points_.ncols(); }

  /// Check that a query has the same dimension as the point cloud.
  void validate_query(const Vector<double>& query) const;

};

}  // namespace svmp

#endif
