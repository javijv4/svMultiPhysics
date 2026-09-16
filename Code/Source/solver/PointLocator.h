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
 * @brief Result of an element-location query.
 *
 * Nearest-centroid queries set @ref location to the element centroid.
 */
struct ElementLocationResult {
  /// Index of the matched element, or -1 if unset.
  int element_index = -1;

  /// Euclidean distance associated with the match.
  double distance = 0.0;

  /// Location associated with the match (element centroid for nearest-centroid).
  Vector<double> location;
};

/**
 * @brief Spatial-query helper over a point cloud or mesh.
 *
 * The locator owns a copy of the supplied point coordinates and, when provided,
 * element connectivity. Element centroids are precomputed for mesh locators.
 * The current implementation uses a deterministic linear scan; a spatial index
 * can be introduced later without changing the public query interface.
 */
class PointLocator {
 public:
  /**
   * @brief Default constructor.
   */
  PointLocator() = default;

  /**
   * @brief Construct a locator from a point cloud.
   *
   * @param[in] points Point coordinates, with one column per point.
   */
  explicit PointLocator(const Array<double>& points);

  /**
   * @brief Construct a locator from a mesh.
   *
   * Copies @p points and @p connectivity and precomputes element centroids.
   *
   * @param[in] points Point coordinates, with one column per point.
   * @param[in] connectivity Element connectivity, with one column per element
   *   and 0-based point indices.
   */
  PointLocator(const Array<double>& points, const Array<int>& connectivity);

  /**
   * @brief Replace stored point coordinates and rebuild centroids if meshed.
   *
   * Connectivity is unchanged. The new point array must match the existing
   * dimension and number of points.
   *
   * @param[in] points Updated point coordinates, with one column per point.
   */
  void update_points(const Array<double>& points);

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

  /**
   * @brief Find the element whose centroid is nearest to a query.
   *
   * Requires connectivity. Distance ties keep the first visited candidate.
   *
   * @param[in] query Query coordinates. Its size must match the point-cloud
   *   dimension.
   * @return The nearest element centroid, or nullopt if the locator has no
   *   elements.
   */
  std::optional<ElementLocationResult> find_nearest_element_centroid(
      const Vector<double>& query) const;

 private:
  /// Point coordinates, with one column per point.
  Array<double> points_;

  /// Element connectivity, with one column per element, or empty.
  Array<int> connectivity_;

  /// Element centroids, with one column per element, or empty.
  Array<double> centroids_;

  /// Return the spatial dimension of the point cloud.
  int dimension() const { return points_.nrows(); }

  /// Return the number of points in the point cloud.
  int number_of_points() const { return points_.ncols(); }

  /// Return the number of stored elements, or 0 if there is no mesh.
  int number_of_elements() const { return connectivity_.ncols(); }

  /// Check that a query has the same dimension as the point cloud.
  void validate_query(const Vector<double>& query) const;
};

}  // namespace svmp

#endif
