#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /src/ares_print_analyzer/analysis/get_histogram.py
# Project: ARES-Print-Analyzer
# Created Date: Friday, October 24th 2025, 10:04:29 am
# Author(s): Graig Ganiano, Nick Kleiner, Arthur W. N. Sloan
# -----
# MIT License
# 
# Copyright (c) 2025 AFRL-ARES
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
###

import numpy as np
from scipy.spatial.distance import pdist
def get_histogram(contourPts):
  # Optmized by Nick Kleiner
  # 1. Squeeze the contour points into an (N, 2) array.
  # contourPts from cv2.findContours has shape (N, 1, 2)
  points = np.squeeze(contourPts)
  n_points = len(points)

  if n_points < 2:
      raise Warning("Not Enough Contour Points")
      return np.zeros((n_points, 60))

  # 2. Use scipy.spatial.distance.pdist for a highly optimized way
  # to calculate all pairwise distances. This replaces the first O(N^2) loop.
  # pdist returns a "condensed" distance matrix (a 1D array).
  all_distances = pdist(points)

  # We need to handle cases where distance is zero.
  log_distances = np.log(np.maximum(1e-5, all_distances))

  maxLogDistance = np.max(log_distances)
  minLogDistance = np.max([0.0, np.min(log_distances)]) # Ensure min is not negative

  radialBound = maxLogDistance + (maxLogDistance - minLogDistance) * 0.01
  intervalSize = radialBound / 5.0
  angleSize = np.pi / 6.0

  # 3. Use NumPy broadcasting to calculate all pairwise differences at once.
  # This is the core of the vectorization for the main histogram calculation.
  # points[:, np.newaxis, :] -> shape (N, 1, 2)
  # points[np.newaxis, :, :] -> shape (1, N, 2)
  # The difference is an (N, N, 2) array of all (dx, dy) pairs.
  diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]

  # 4. Calculate angles and distances for all pairs simultaneously.
  # These operations now act on entire (N, N) matrices.
  angles = np.arctan2(diffs[:, :, 1], diffs[:, :, 0])
  distances = np.sqrt(diffs[:, :, 0]**2 + diffs[:, :, 1]**2)

  log_distances_matrix = np.log(np.maximum(1e-5, distances))

  # 5. Calculate the bin for every pair at once.
  angle_bins = (angles / angleSize).astype(int)
  distance_bins = (log_distances_matrix / intervalSize).astype(int)

  # Combine into a single index for the 60-bin histogram.
  indices = distance_bins * 12 + angle_bins

  # 6. Efficiently populate the histogram.
  # We create a mask to ignore invalid indices and self-comparisons.
  histogram = np.zeros((n_points, 60), dtype=int)
  valid_mask = (indices >= 0) & (indices < 60)
  np.fill_diagonal(valid_mask, False) # Ignore point-to-self comparisons

  # Loop through each point and use the super-fast np.bincount
  # to count the occurrences of each bin index for that point.
  for i in range(n_points):
      # Get the valid indices for this row
      row_indices = indices[i, valid_mask[i]]
      # np.bincount is perfect for this task.
      counts = np.bincount(row_indices, minlength=60)
      histogram[i, :] = counts[:60] # Ensure we don't exceed 60 bins

  return histogram

# Graig's original Analysis code
# def dist(p1, p2):
#   return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# def angle(p1, p2):
#   ydif = p2[1] - p1[1]
#   xdif = p2[0] - p1[0]
#   theta = math.atan2(ydif, xdif)

#   return theta

# def get_histogram(contourPts):
#   points = [pt[0] for pt in contourPts]

#   maxLogDistance = -float('inf')
#   minLogDistance = float('inf')

#   for i in range(len(points)):
#       for j in range(i + 1, len(points)):
#           distance = dist(points[i], points[j]) 
#           logDistance = math.log(max(1e-5,distance))
#           if logDistance > maxLogDistance:
#               maxLogDistance = logDistance
#           if logDistance < minLogDistance:
#               minLogDistance = max(0.0, logDistance)

#   radialBound = maxLogDistance + (maxLogDistance - minLogDistance) * 0.01
#   intervalSize = radialBound / 5.0
#   angleSize = math.pi / 6.0


#   histogram = [[0 for _ in range(60)] for _ in range(len(points))]


#   for i in range(len(points)):
#       for j in range(len(points)):
#           if i != j:
#               ang = angle(points[i], points[j])
#               angleBin = int(ang / angleSize)
#               distance = dist(points[i], points[j]) 
#               distance = max(0.0, math.log(max(1e-5,distance)))
#               distanceBin = int(distance / intervalSize)
#               index = distanceBin * 12 + angleBin
#               if 0 <= index < 60:
#                   histogram[i][index] += 1

#   return histogram
