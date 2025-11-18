#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /src/ares_print_analyzer/analysis/get_chi_statistic.py
# Project: ARES-Print-Analyzer
# Created Date: Friday, October 24th 2025, 9:40:21 am
# Author(s): Graig Ganitano, Nick Kleiner, Arthur W. N. Sloan
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


def get_chi_statistic(histogram1: list | np.ndarray,
                      histogram2: list | np.ndarray,
                      outlier_threshold: float = 1.2) -> np.ndarray:
  # Optimized Routine by Nick Kleiner 
  OUTLIER_THRESHOLD = outlier_threshold
  EPSILON = 1e-16  # A small number to prevent division by zero

  # 1. Convert lists to NumPy arrays for vectorized operations.
  h1 = np.atleast_1d(histogram1).astype(np.float32)
  h2 = np.atleast_1d(histogram2).astype(np.float32)

  size1, _ = h1.shape
  size2, _ = h2.shape
  size = max(size1, size2)

  # 2. Initialize the final stats matrix with the outlier value.
  # This handles the padding logic from your original function upfront.
  stats = np.full((size, size), OUTLIER_THRESHOLD)

  # If either histogram is empty, we can't compute, so return the outlier matrix.
  if size1 == 0 or size2 == 0:
      raise Warning("One of the histograms is empty, cannot compute chi statistic")
      return stats

  # 3. Use broadcasting to compute all pairwise differences and sums at once.
  # h1[:, np.newaxis, :] expands h1 to shape (size1, 1, 60)
  # h2[np.newaxis, :, :] expands h2 to shape (1, size2, 60)
  # NumPy then broadcasts them to a common shape of (size1, size2, 60).
  diff = h1[:, np.newaxis, :] - h2[np.newaxis, :, :]
  sum_ = h1[:, np.newaxis, :] + h2[np.newaxis, :, :]

  # 4. Perform the Chi-squared calculation in a single, vectorized step.
  # We sum along the last axis (axis=2), which is the 60 histogram bins.
  # This collapses the (size1, size2, 60) matrix into a (size1, size2) result.
  chi_sq_matrix = np.sum((diff ** 2) / (sum_ + EPSILON), axis=2) / 2

  # 5. Place the computed results into the top-left corner of the padded matrix.
  stats[:size1, :size2] = chi_sq_matrix
  # The scipy implementation of the linear sum assignment problem can handle non-square cost matrices, so we nolonger need to pad
  stats = chi_sq_matrix
  return chi_sq_matrix

# Original Routine by Graig Ganitano

# def get_chi_statistic(histogram1, histogram2):
#   OUTLIER_THRESHOLD = 1.2
#   size1, size2 = len(histogram1), len(histogram2)
#   size = max(size1, size2)
#   stats = np.zeros((size, size))  

#   for i in range(size):
#       for j in range(size):
#           if i >= size1 or j >= size2:
#               stats[i, j] = OUTLIER_THRESHOLD
#               continue

#           summation = 0
#           for k in range(60):  
#               diff = histogram1[i][k] - histogram2[j][k] if i < size1 and j < size2 else 0
#               sum_ = histogram1[i][k] + histogram2[j][k] if i < size1 and j < size2 else 1
#               if sum_ != 0:  
#                   summation += (diff * diff) / sum_
#           stats[i, j] = summation / 2

#   return stats