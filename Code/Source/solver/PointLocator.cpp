// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "PointLocator.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace svmp {

namespace {

double squared_distance(const Vector<double>& first, const Vector<double>& second) {
  double value = 0.0;
  for (int dimension = 0; dimension < first.size(); ++dimension) {
    const double difference = first(dimension) - second(dimension);
    value += difference * difference;
  }
  return value;
}

void validate_points(const Array<double>& points) {
  if (points.nrows() <= 0 || points.ncols() <= 0) {
    throw std::invalid_argument("PointLocator requires a non-empty point array.");
  }
}

void validate_mesh(const Array<double>& points, const Array<int>& connectivity) {
  validate_points(points);
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
}

Array<double> compute_centroids(const Array<double>& points, const Array<int>& connectivity) {
  Array<double> centroids(points.nrows(), connectivity.ncols());
  for (int element = 0; element < connectivity.ncols(); ++element) {
    for (int dimension_index = 0; dimension_index < points.nrows(); ++dimension_index) {
      double coordinate = 0.0;
      for (int node = 0; node < connectivity.nrows(); ++node) {
        coordinate += points(dimension_index, connectivity(node, element));
      }
      centroids(dimension_index, element) = coordinate / connectivity.nrows();
    }
  }
  return centroids;
}

}  // namespace

PointLocator::PointLocator(const Array<double>& points) {
  validate_points(points);
  points_ = points;
}

PointLocator::PointLocator(const Array<double>& points, const Array<int>& connectivity) {
  validate_mesh(points, connectivity);
  points_ = points;
  connectivity_ = connectivity;
  centroids_ = compute_centroids(points_, connectivity_);
}

void PointLocator::update_points(const Array<double>& points) {
  validate_points(points);
  if (points.nrows() != dimension() || points.ncols() != number_of_points()) {
    throw std::invalid_argument(
        "PointLocator update_points requires matching point dimensions.");
  }
  points_ = points;
  if (number_of_elements() > 0) {
    centroids_ = compute_centroids(points_, connectivity_);
  }
}

void PointLocator::validate_query(const Vector<double>& query) const {
  if (query.size() != dimension()) {
    throw std::invalid_argument("PointLocator query dimension does not match point dimension.");
  }
}

std::optional<NearestPointResult> PointLocator::find_nearest_neighbor(
    const Vector<double>& query, std::optional<int> exclude_index) const {
  validate_query(query);
  if (exclude_index.has_value() &&
      (*exclude_index < 0 || *exclude_index >= number_of_points())) {
    throw std::invalid_argument("PointLocator exclude index is out of range.");
  }

  double minimum_squared_distance = std::numeric_limits<double>::max();
  int nearest_index = -1;

  for (int point_index = 0; point_index < number_of_points(); ++point_index) {
    if (exclude_index.has_value() && point_index == *exclude_index) {
      continue;
    }
    const double candidate_squared_distance =
        squared_distance(query, points_.col(point_index));
    if (candidate_squared_distance < minimum_squared_distance) {
      minimum_squared_distance = candidate_squared_distance;
      nearest_index = point_index;
    }
  }

  if (nearest_index == -1) {
    return std::nullopt;
  }
  return NearestPointResult{nearest_index, std::sqrt(minimum_squared_distance)};
}

std::optional<ElementLocationResult> PointLocator::find_nearest_element_centroid(
    const Vector<double>& query) const {
  validate_query(query);
  if (number_of_elements() == 0) {
    throw std::logic_error("PointLocator element query requires connectivity.");
  }

  double minimum_squared_distance = std::numeric_limits<double>::max();
  int nearest_index = -1;
  for (int element_index = 0; element_index < number_of_elements(); ++element_index) {
    const double candidate_squared_distance =
        squared_distance(query, centroids_.col(element_index));
    if (candidate_squared_distance < minimum_squared_distance) {
      minimum_squared_distance = candidate_squared_distance;
      nearest_index = element_index;
    }
  }

  if (nearest_index == -1) {
    return std::nullopt;
  }

  ElementLocationResult result;
  result.element_index = nearest_index;
  result.distance = std::sqrt(minimum_squared_distance);
  result.location = centroids_.col(nearest_index);
  return result;
}

}  // namespace svmp
