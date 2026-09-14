// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef POINT_LOCATOR_H
#define POINT_LOCATOR_H

#include "Array.h"
#include "Vector.h"

#include <optional>
#include <vector>

namespace svmp {

/**
 * @brief Result of a nearest-point query.
 */
struct NearestPointResult {
  /// Index of the matched point in the locator point array, or -1 if unset.
  int point_index = -1;

  /// Euclidean distance from the query to the matched point.
  double distance = 0.0;
};

/**
 * @brief Result of an element-location query.
 *
 * The populated fields depend on the query. Containing-element queries set
 * @ref location to the query point and @ref shape_functions to the barycentric
 * weights of that point in the matched element. Nearest-centroid queries set
 * @ref location to the element centroid and leave @ref shape_functions empty.
 */
struct ElementLocationResult {
  /// Index of the matched element, or -1 if unset.
  int element_index = -1;

  /// Euclidean distance associated with the match, when the query reports one.
  double distance = 0.0;

  /// Location associated with the match. See the class documentation.
  Vector<double> location;

  /// Barycentric weights of the query in the matched element, when computed.
  Vector<double> shape_functions;
};

/**
 * @brief Spatial-query helper over a point cloud, optionally with a mesh.
 *
 * This class owns a copy of the point coordinates it is constructed from, and
 * optionally a copy of element connectivity. It answers nearest-point,
 * containment and box queries against that data. The current implementation
 * uses deterministic linear scans so that results do not depend on a spatial
 * index; a future indexed backend can replace the private implementation
 * without changing callers.
 *
 * Point coordinates are stored as an @c Array. Connectivity, when provided, 
 * is stored as an @c Array.
 */
class PointLocator {
 public:
  /**
   * @brief Construct a locator from a point cloud.
   *
   * Copies @p points and leaves connectivity empty. The resulting locator
   * can answer point queries but not element queries.
   *
   * @param[in] points Point coordinates, with one column per point.
   */
  PointLocator(const Array<double>& points);

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
   * @brief Replace the stored point cloud and clear connectivity.
   *
   * @param[in] points Point coordinates, with one column per point.
   */
  void update_points(const Array<double>& points);

  /**
   * @brief Replace the stored mesh and rebuild element centroids.
   *
   * @param[in] points Point coordinates, with one column per point.
   * @param[in] connectivity Element connectivity, with one column per element
   *   and 0-based point indices.
   */
  void update_mesh(const Array<double>& points, const Array<int>& connectivity);

  /**
   * @brief Find the nearest stored point to a query.
   *
   * Distance ties keep the first visited candidate. Pass @p excluded_point to
   * skip a known point, for example when searching for a neighbor of a point
   * that is itself in the cloud.
   *
   * @param[in] query Query coordinates. The size must match @ref dimension.
   * @param[in] candidates Optional subset of point indices to search. Null
   *   searches every stored point.
   * @param[in] excluded_point Point index to skip, or -1 to skip none.
   * @return The nearest point, or an empty optional if no candidate remains.
   */
  std::optional<NearestPointResult> find_nearest_neighbor(
      const Vector<double>& query, const std::vector<int>* candidates = nullptr,
      int excluded_point = -1) const;

  /**
   * @brief Find the first stored point within a given distance of a query.
   *
   * Candidates are visited in order. The first point whose Euclidean distance
   * is at most @p distance is returned.
   *
   * @param[in] query Query coordinates. The size must match @ref dimension.
   * @param[in] distance Inclusive search radius. A negative value yields an
   *   empty optional.
   * @param[in] candidates Optional subset of point indices to search. Null
   *   searches every stored point.
   * @param[in] excluded_point Point index to skip, or -1 to skip none.
   * @return The first matching point, or an empty optional if none lies
   *   within @p distance.
   */
  std::optional<NearestPointResult> find_first_within_distance(
      const Vector<double>& query, double distance,
      const std::vector<int>* candidates = nullptr,
      int excluded_point = -1) const;

  /**
   * @brief Find the element whose centroid is nearest to a query.
   *
   * Requires connectivity. Distance ties keep the first visited candidate.
   *
   * @param[in] query Query coordinates. The size must match @ref dimension.
   * @param[in] candidates Optional subset of element indices to search. Null
   *   searches every stored element.
   * @return The nearest element centroid, or an empty optional if no
   *   candidate remains.
   */
  std::optional<ElementLocationResult> find_nearest_element_centroid(
      const Vector<double>& query,
      const std::vector<int>* candidates = nullptr) const;

  /**
   * @brief Find the first element that geometrically contains a query.
   *
   * Requires simplex connectivity. Each candidate is first rejected by an
   * axis-aligned bounding box, then tested with barycentric coordinates.
   * Candidates are visited in order; the first containing element is
   * returned.
   *
   * @param[in] query Query coordinates. The size must match @ref dimension.
   * @param[in] candidates Optional subset of element indices to search. Null
   *   searches every stored element.
   * @param[in] include_boundary If true, points on the element boundary are
   *   treated as inside.
   * @return The containing element and barycentric weights, or an empty
   *   optional if no candidate contains the query.
   */
  std::optional<ElementLocationResult> find_containing_element_geometric(
      const Vector<double>& query, const std::vector<int>* candidates = nullptr,
      bool include_boundary = true) const;

  /**
   * @brief Find the first element whose barycentric weights contain a query.
   *
   * Requires simplex connectivity. An element contains the query if every
   * barycentric weight lies in @f$[-t, 1 + t]@f$, where @f$t@f$ is
   * @p tolerance. Candidates are visited in order; the first containing
   * element is returned.
   *
   * @param[in] query Query coordinates. The size must match @ref dimension.
   * @param[in] candidates Optional subset of element indices to search. Null
   *   searches every stored element.
   * @param[in] tolerance Inclusive tolerance applied to each barycentric
   *   weight.
   * @return The containing element and barycentric weights, or an empty
   *   optional if no candidate contains the query.
   */
  std::optional<ElementLocationResult> find_containing_element_barycentric(
      const Vector<double>& query, const std::vector<int>* candidates = nullptr,
      double tolerance = 1.0e-14) const;

  /**
   * @brief Find the stored points that lie in a closed axis-aligned box.
   *
   * @param[in] minimum Inclusive lower corner of the box.
   * @param[in] maximum Inclusive upper corner of the box.
   * @param[in] candidates Optional subset of point indices to search. Null
   *   searches every stored point.
   * @return Indices of the points inside the box, in visit order.
   */
  std::vector<int> find_points_in_box(const Vector<double>& minimum,
                                      const Vector<double>& maximum,
                                      const std::vector<int>* candidates = nullptr) const;

  /**
   * @brief Return whether a point lies in a closed axis-aligned box.
   *
   * @param[in] point Point to test.
   * @param[in] minimum Inclusive lower corner of the box.
   * @param[in] maximum Inclusive upper corner of the box.
   * @return True if every coordinate of @p point lies between the
   *   corresponding coordinates of @p minimum and @p maximum.
   */
  static bool point_in_box(const Vector<double>& point, const Vector<double>& minimum,
                           const Vector<double>& maximum);

  /**
   * @brief Return whether a point lies in a closed ball.
   *
   * @param[in] point Point to test.
   * @param[in] center Center of the ball.
   * @param[in] radius Inclusive radius. Must be non-negative.
   * @return True if the Euclidean distance from @p point to @p center is at
   *   most @p radius.
   */
  static bool point_in_sphere(const Vector<double>& point, const Vector<double>& center,
                              double radius);

 private:
  /// Point coordinates, with one column per point.
  Array<double> points_;

  /// Element connectivity, with one column per element, or empty.
  Array<int> connectivity_;

  /// Element centroids, with one column per element, or empty.
  Array<double> centroids_;

  /**
   * @brief Spatial dimension of the stored points.
   *
   * @return Number of rows in the point array.
   */
  int dimension() const { return points_.nrows(); }

  /**
   * @brief Number of stored points.
   *
   * @return Number of columns in the point array.
   */
  int number_of_points() const { return points_.ncols(); }

  /**
   * @brief Number of stored elements.
   *
   * @return Number of columns in the connectivity array, or 0 if the locator
   *   has no mesh.
   */
  int number_of_elements() const { return connectivity_.ncols(); }

  /**
   * @brief Check that a query has the same dimension as the stored points.
   *
   * @param[in] query Query coordinates to validate.
   */
  void validate_query(const Vector<double>& query) const;

  /**
   * @brief Recompute element centroids from the stored mesh.
   */
  void rebuild_centroids();

  /**
   * @brief Resolve the element indices to visit for a query.
   *
   * @param[in] candidates Optional subset of element indices. Null selects
   *   every stored element.
   * @return The indices to visit, in order.
   */
  std::vector<int> element_indices(const std::vector<int>* candidates) const;

  /**
   * @brief Resolve the point indices to visit for a query.
   *
   * @param[in] candidates Optional subset of point indices. Null selects
   *   every stored point.
   * @return The indices to visit, in order.
   */
  std::vector<int> point_indices(const std::vector<int>* candidates) const;

  /**
   * @brief Return whether a simplex element geometrically contains a query.
   *
   * @param[in] query Query coordinates.
   * @param[in] element_index Element to test.
   * @param[in] include_boundary If true, points on the element boundary are
   *   treated as inside.
   * @return True if the element contains the query.
   */
  bool geometric_contains(const Vector<double>& query, int element_index,
                          bool include_boundary) const;

  /**
   * @brief Compute barycentric coordinates of a query in a simplex element.
   *
   * @param[in] query Query coordinates.
   * @param[in] element_index Element in which the coordinates are computed.
   * @return The barycentric weights, or an empty optional if the element is
   *   degenerate.
   */
  std::optional<Vector<double>> barycentric_coordinates(const Vector<double>& query,
                                                         int element_index) const;
};

}  // namespace svmp

#endif
