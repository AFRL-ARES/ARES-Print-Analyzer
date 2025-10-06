from PyAres import AresAnalyzerService, Analysis, AnalysisRequest, AresDataType
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
import numpy as np
import math
import cv2

def convert_image_bytes_to_ndarray(image_bytes) -> np.ndarray:
  nparr = np.frombuffer(image_bytes, np.uint8)
  img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  return img_np

def dist(p1, p2):
  return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def angle(p1, p2):
  ydif = p2[1] - p1[1]
  xdif = p2[0] - p1[0]
  theta = math.atan2(ydif, xdif)

  return theta

def get_chi_statistic_optimized(histogram1, histogram2):
  OUTLIER_THRESHOLD = 1.2
  EPSILON = 1e-9  # A small number to prevent division by zero

  # 1. Convert lists to NumPy arrays for vectorized operations.
  h1 = np.array(histogram1, dtype=np.float32)
  h2 = np.array(histogram2, dtype=np.float32)

  size1, _ = h1.shape
  size2, _ = h2.shape
  size = max(size1, size2)

  # 2. Initialize the final stats matrix with the outlier value.
  # This handles the padding logic from your original function upfront.
  stats = np.full((size, size), OUTLIER_THRESHOLD)

  # If either histogram is empty, we can't compute, so return the outlier matrix.
  if size1 == 0 or size2 == 0:
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

  return stats

def get_chi_statistic(histogram1, histogram2):
  OUTLIER_THRESHOLD = 1.2
  size1, size2 = len(histogram1), len(histogram2)
  size = max(size1, size2)
  stats = np.zeros((size, size))  

  for i in range(size):
      for j in range(size):
          if i >= size1 or j >= size2:
              stats[i, j] = OUTLIER_THRESHOLD
              continue

          summation = 0
          for k in range(60):  
              diff = histogram1[i][k] - histogram2[j][k] if i < size1 and j < size2 else 0
              sum_ = histogram1[i][k] + histogram2[j][k] if i < size1 and j < size2 else 1
              if sum_ != 0:  
                  summation += (diff * diff) / sum_
          stats[i, j] = summation / 2

  return stats

def get_histogram_optimized(contourPts):
  # 1. Squeeze the contour points into an (N, 2) array.
  # contourPts from cv2.findContours has shape (N, 1, 2)
  points = np.squeeze(contourPts)
  n_points = len(points)

  if n_points < 2:
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
  angleSize = math.pi / 6.0

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

def get_histogram(contourPts):
  points = [pt[0] for pt in contourPts]

  maxLogDistance = -float('inf')
  minLogDistance = float('inf')

  for i in range(len(points)):
      for j in range(i + 1, len(points)):
          distance = dist(points[i], points[j]) 
          logDistance = math.log(max(1e-5,distance))
          if logDistance > maxLogDistance:
              maxLogDistance = logDistance
          if logDistance < minLogDistance:
              minLogDistance = max(0.0, logDistance)

  radialBound = maxLogDistance + (maxLogDistance - minLogDistance) * 0.01
  intervalSize = radialBound / 5.0
  angleSize = math.pi / 6.0


  histogram = [[0 for _ in range(60)] for _ in range(len(points))]


  for i in range(len(points)):
      for j in range(len(points)):
          if i != j:
              ang = angle(points[i], points[j])
              angleBin = int(ang / angleSize)
              distance = dist(points[i], points[j]) 
              distance = max(0.0, math.log(max(1e-5,distance)))
              distanceBin = int(distance / intervalSize)
              index = distanceBin * 12 + angleBin
              if 0 <= index < 60:
                  histogram[i][index] += 1

  return histogram

def analyze(request: AnalysisRequest) -> Analysis:
  image_bytes: bytes = request.inputs["Image"]
  base_image: str = request.settings["Base Image Path"]
  image_numpy_array = convert_image_bytes_to_ndarray(image_bytes)
  synthetic_image = cv2.imread(base_image)
  synthetic_image_gray = cv2.cvtColor(synthetic_image, cv2.COLOR_BGR2GRAY)
  ret, syn_mask = cv2.threshold(synthetic_image_gray, 40, 255, cv2.THRESH_BINARY)
  contours, hierarchy = cv2.findContours(syn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  largest_contour = None
  max_area = 0

  for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
      max_area = area
      largest_contour = contour
  
  hsv = cv2.cvtColor(image_numpy_array, cv2.COLOR_BGR2HSV)
  hsv_mask = cv2.inRange(hsv, (95, 95, 95), (179, 255, 255))

  largest_contour_camera = None
  max_area_camera = 0

  contours, hierarchy = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  if not contours:
    return 0
  
  for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area_camera:
      max_area_camera = area
      largest_contour_camera = contour
  
  hist_syn = get_histogram_optimized(largest_contour)
  hist_camera = get_histogram_optimized(largest_contour_camera)
  stats = get_chi_statistic_optimized(hist_syn, hist_camera)
  row_ind, col_ind = linear_sum_assignment(stats)

  score = stats[row_ind, col_ind].sum()/len(row_ind)
  #D_MAX = 2.0
  #D_clipped = min(score, D_MAX)

  #normalized_score = 10 * max(1.0 - (D_clipped / D_MAX))
  analysis = Analysis(score, True)
  return analysis

if __name__ == "__main__":
  print("PyAres Print Analyzer")
  description = "A PyAres implementation of Graig Ganitano's 3D Printing Analyzer"
  analyzer = AresAnalyzerService(analyze, "Print Analyzer", "1.0.0", description)

  analyzer.add_analysis_parameter("Image", AresDataType.BYTE_ARRAY)
  analyzer.add_setting("Base Image Path", AresDataType.STRING)
  analyzer.start()