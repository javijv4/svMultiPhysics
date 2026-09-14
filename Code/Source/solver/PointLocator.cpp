// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "PointLocator.h"
#include "mat_fun.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace svmp {

namespace {

double squared_distance(const Vector<double>& first, const Vector<double>& second) {
  double value = 0.0;
  for (int i = 0; i < first.size(); ++i) {
    const double difference = first(i) - second(i);
    value += difference * difference;
  }
  return value;
}

}  // namespace

PointLocator::PointLocator(const Array<double>& points) { update_points(points); }

PointLocator::PointLocator(const Array<double>& points, const Array<int>& connectivity) {
  update_mesh(points, connectivity);
}

void PointLocator::update_points(const Array<double>& points) {
  if (points.nrows() <= 0 || points.ncols() <= 0) {
    throw std::invalid_argument("PointLocator requires a non-empty point array.");
  }
  points_ = points;
}

void PointLocator::update_mesh(const Array<double>& points, const Array<int>& connectivity) {
  if (points.nrows() <= 0 || points.ncols() <= 0) {
    throw std::invalid_argument("PointLocator requires a non-empty point array.");
  }
  if (connectivity.nrows() <= 0 || connectivity.ncols() <= 0) {
    throw std::invalid_argument("PointLocator requires non-empty element connectivity.");
  }
  for (int element = 0; element < connectivity.ncols(); ++element) {
    for (int node = 0; node < connectivity.nrows(); ++node) {
      const int point_index = connectivity(node, element);
      if (point_index < 0 || point_index >= points.ncols()) {
        throw std::invalid_argument("PointLocator connectivity contains an invalid point index.");
      }
    }
  }
  points_ = points;
  connectivity_ = connectivity;
  rebuild_centroids();
}

void PointLocator::validate_query(const Vector<double>& query) const {
  if (query.size() != dimension()) {
    throw std::invalid_argument("PointLocator query dimension does not match point dimension.");
  }
}

void PointLocator::rebuild_centroids() {
  centroids_.resize(points_.nrows(), connectivity_.ncols());
  for (int element = 0; element < connectivity_.ncols(); ++element) {
    for (int dimension_index = 0; dimension_index < points_.nrows(); ++dimension_index) {
      double coordinate = 0.0;
      for (int node = 0; node < connectivity_.nrows(); ++node) {
        coordinate += points_(dimension_index, connectivity_(node, element));
      }
      centroids_(dimension_index, element) = coordinate / connectivity_.nrows();
    }
  }
}

std::vector<int> PointLocator::point_indices(const std::vector<int>* candidates) const {
  if (candidates != nullptr) {
    for (const int index : *candidates) {
      if (index < 0 || index >= number_of_points()) {
        throw std::invalid_argument("PointLocator point candidate index is out of range.");
      }
    }
    return *candidates;
  }
  std::vector<int> indices(number_of_points());
  for (int index = 0; index < number_of_points(); ++index) indices[index] = index;
  return indices;
}

std::vector<int> PointLocator::element_indices(const std::vector<int>* candidates) const {
  if (number_of_elements() == 0) {
    throw std::logic_error("PointLocator element query requires connectivity.");
  }
  if (candidates != nullptr) {
    for (const int index : *candidates) {
      if (index < 0 || index >= number_of_elements()) {
        throw std::invalid_argument("PointLocator element candidate index is out of range.");
      }
    }
    return *candidates;
  }
  std::vector<int> indices(number_of_elements());
  for (int index = 0; index < number_of_elements(); ++index) indices[index] = index;
  return indices;
}

std::optional<NearestPointResult> PointLocator::find_nearest_neighbor(
    const Vector<double>& query, const std::vector<int>* candidates, int excluded_point) const {
  validate_query(query);
  double minimum_squared_distance = std::numeric_limits<double>::max();
  int nearest_index = -1;
  for (const int point_index : point_indices(candidates)) {
    if (point_index == excluded_point) continue;
    const double candidate_squared_distance = squared_distance(query, points_.col(point_index));
    if (candidate_squared_distance < minimum_squared_distance) {
      minimum_squared_distance = candidate_squared_distance;
      nearest_index = point_index;
    }
  }
  if (nearest_index == -1) return std::nullopt;
  return NearestPointResult{nearest_index, std::sqrt(minimum_squared_distance)};
}

std::optional<NearestPointResult> PointLocator::find_first_within_distance(
    const Vector<double>& query, const double distance, const std::vector<int>* candidates,
    const int excluded_point) const {
  validate_query(query);
  if (distance < 0.0) return std::nullopt;
  const double maximum_squared_distance = distance * distance;
  for (const int point_index : point_indices(candidates)) {
    if (point_index == excluded_point) continue;
    const double candidate_squared_distance = squared_distance(query, points_.col(point_index));
    if (candidate_squared_distance <= maximum_squared_distance) {
      return NearestPointResult{point_index, std::sqrt(candidate_squared_distance)};
    }
  }
  return std::nullopt;
}

std::optional<ElementLocationResult> PointLocator::find_nearest_element_centroid(
    const Vector<double>& query, const std::vector<int>* candidates) const {
  validate_query(query);
  double minimum_squared_distance = std::numeric_limits<double>::max();
  int nearest_index = -1;
  for (const int element_index : element_indices(candidates)) {
    const double candidate_squared_distance = squared_distance(query, centroids_.col(element_index));
    if (candidate_squared_distance < minimum_squared_distance) {
      minimum_squared_distance = candidate_squared_distance;
      nearest_index = element_index;
    }
  }
  if (nearest_index == -1) return std::nullopt;
  ElementLocationResult result;
  result.element_index = nearest_index;
  result.distance = std::sqrt(minimum_squared_distance);
  result.location = centroids_.col(nearest_index);
  return result;
}

bool PointLocator::point_in_box(const Vector<double>& point, const Vector<double>& minimum,
                                const Vector<double>& maximum) {
  if (point.size() != minimum.size() || point.size() != maximum.size()) {
    throw std::invalid_argument("PointLocator box dimensions do not match.");
  }
  for (int i = 0; i < point.size(); ++i) {
    if (minimum(i) > maximum(i)) {
      throw std::invalid_argument("PointLocator box minimum exceeds maximum.");
    }
    if (point(i) < minimum(i) || point(i) > maximum(i)) return false;
  }
  return true;
}

bool PointLocator::point_in_sphere(const Vector<double>& point, const Vector<double>& center,
                                   const double radius) {
  if (radius < 0.0) throw std::invalid_argument("PointLocator sphere radius is negative.");
  if (point.size() != center.size()) {
    throw std::invalid_argument("PointLocator sphere dimensions do not match.");
  }
  return squared_distance(point, center) <= radius * radius;
}

std::vector<int> PointLocator::find_points_in_box(const Vector<double>& minimum,
                                                   const Vector<double>& maximum,
                                                   const std::vector<int>* candidates) const {
  if (minimum.size() != dimension() || maximum.size() != dimension()) {
    throw std::invalid_argument("PointLocator box dimension does not match point dimension.");
  }
  std::vector<int> result;
  for (const int point_index : point_indices(candidates)) {
    if (point_in_box(points_.col(point_index), minimum, maximum)) result.push_back(point_index);
  }
  return result;
}

std::optional<Vector<double>> PointLocator::barycentric_coordinates(
    const Vector<double>& query, const int element_index) const {
  const int dimensions = dimension();
  const int nodes = dimensions + 1;
  if (connectivity_.nrows() != nodes) {
    throw std::logic_error("PointLocator barycentric containment requires simplex connectivity.");
  }

  // Build the augmented system
  Array<double> Amat(nodes, nodes);
  Amat = 1.0;
  for (int node = 0; node < nodes; ++node) {
    const int point_index = connectivity_(node, element_index);
    for (int row = 0; row < dimensions; ++row) {
      Amat(row, node) = points_(row, point_index);
    }
  }
  // Check if the system is singular.
  if (utils::is_zero(std::fabs(mat_fun::mat_det(Amat, nodes)))) return std::nullopt;

  // Build the right-hand side of the system.
  Vector<double> rhs(nodes);
  for (int row = 0; row < dimensions; ++row) rhs(row) = query(row);
  rhs(dimensions) = 1.0;

  // Solve the system and return the barycentric coordinates.
  return mat_fun::mat_mul(mat_fun::mat_inv(Amat, nodes), rhs);
}

bool PointLocator::geometric_contains(const Vector<double>& query, const int element_index,
                                      const bool include_boundary) const {
  const int dimensions = dimension();
  if (connectivity_.nrows() != dimensions + 1) {
    throw std::logic_error("PointLocator geometric containment requires simplex connectivity.");
  }
  Vector<double> minimum(dimensions);
  Vector<double> maximum(dimensions);
  for (int dimension_index = 0; dimension_index < dimensions; ++dimension_index) {
    minimum(dimension_index) = std::numeric_limits<double>::max();
    maximum(dimension_index) = std::numeric_limits<double>::lowest();
    for (int node = 0; node < connectivity_.nrows(); ++node) {
      const double coordinate = points_(dimension_index, connectivity_(node, element_index));
      minimum(dimension_index) = std::min(minimum(dimension_index), coordinate);
      maximum(dimension_index) = std::max(maximum(dimension_index), coordinate);
    }
    const double tolerance = std::max((maximum(dimension_index) - minimum(dimension_index)) * 1.0e-3, 1.0e-12);
    minimum(dimension_index) -= tolerance;
    maximum(dimension_index) += tolerance;
  }
  if (!point_in_box(query, minimum, maximum)) return false;
  const auto weights = barycentric_coordinates(query, element_index);
  if (!weights) return false;
  const double tolerance = include_boundary ? 1.0e-12 : 0.0;
  for (int node = 0; node < weights->size(); ++node) {
    if (include_boundary ? ((*weights)(node) < -tolerance) : ((*weights)(node) <= 0.0)) return false;
  }
  return true;
}

std::optional<ElementLocationResult> PointLocator::find_containing_element_geometric(
    const Vector<double>& query, const std::vector<int>* candidates, const bool include_boundary) const {
  validate_query(query);
  for (const int element_index : element_indices(candidates)) {
    if (!geometric_contains(query, element_index, include_boundary)) continue;
    ElementLocationResult result;
    result.element_index = element_index;
    result.location = query;
    result.shape_functions = *barycentric_coordinates(query, element_index);
    return result;
  }
  return std::nullopt;
}

std::optional<ElementLocationResult> PointLocator::find_containing_element_barycentric(
    const Vector<double>& query, const std::vector<int>* candidates, const double tolerance) const {
  validate_query(query);
  for (const int element_index : element_indices(candidates)) {
    const auto weights = barycentric_coordinates(query, element_index);
    if (!weights) continue;
    bool contains = true;
    for (int node = 0; node < weights->size(); ++node) {
      if ((*weights)(node) < -tolerance || (*weights)(node) > 1.0 + tolerance) {
        contains = false;
        break;
      }
    }
    if (!contains) continue;
    ElementLocationResult result;
    result.element_index = element_index;
    result.location = query;
    result.shape_functions = *weights;
    return result;
  }
  return std::nullopt;
}

}  // namespace svmp
