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

}  // namespace

PointLocator::PointLocator(const Array<double>& points) {
  if (points.nrows() <= 0 || points.ncols() <= 0) {
    throw std::invalid_argument("PointLocator requires a non-empty point array.");
  }
  points_ = points;
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

}  // namespace svmp
