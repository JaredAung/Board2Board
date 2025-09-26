"""
Board2Board – Secondary Preprocessing Module
Author: Jared Aung © 2025 All rights reserved.

Description:
------------
This module performs the **secondary preprocessing stage** of chessboard detection:
1. Detects grid lines on the warped chessboard using Hough Transform.
2. Computes intersections of vertical and horizontal lines.
3. Clusters intersection points to refine accuracy.
4. Sorts clustered intersections into a structured 9×9 grid.
5. Visualizes detected grid squares with chess notation labels.

Additionally:
- Extracts training features (intensity, edge density, gradient, entropy, etc.)
  for machine learning models that predict optimal thresholds.
- Integrates ML models (via `joblib`) to automatically determine thresholds
  if not provided.
"""

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist
from collections import defaultdict
from skimage.measure import shannon_entropy
import joblib

# -------------------- DATA STRUCTURES --------------------
# Metrics/features extracted during secondary Hough Transform
secondary_hough_training_data = {
    "mean_intensity": 0.0,
    "std_intensity": 0.0,
    "edge_density": 0.0,
    "gradient_mean": 0.0,
    "gradient_std": 0.0,
    "entropy": 0.0,
    "aspect_ratio": 0.0,
    "mean_distance": 0.0,
    "std_distance": 0.0,
    "hough_threshold": 0
}

# Metrics/features extracted during secondary clustering
secondary_clustering_data = {
    "num_points": 0,
    "mean_distance": 0.0,
    "std_distance": 0.0,
    "cluster_threshold": 0
}

# -------------------- HELPER FUNCTIONS --------------------
def compute_intersection(rho1, theta1, rho2, theta2):
    """
    Computes the intersection point of two Hough lines (in polar form).

    Args:
        rho1, theta1: Parameters of the first line.
        rho2, theta2: Parameters of the second line.

    Returns:
        (x, y) as integer tuple, or None if lines are nearly parallel.
    """
    a1, b1 = np.cos(theta1), np.sin(theta1)
    a2, b2 = np.cos(theta2), np.sin(theta2)
    A = np.array([[a1, b1], [a2, b2]])
    B = np.array([[rho1], [rho2]])
    if np.abs(np.linalg.det(A)) < 1e-10:
        return None
    intersection = np.linalg.solve(A, B)
    return tuple(np.round(intersection.flatten()).astype(int))

def cluster_points(points, threshold=40):
    """
    Clusters raw intersection points into averaged cluster centers.

    Args:
        points (list): List of (x, y) tuples.
        threshold (int): Distance threshold for clustering.

    Returns:
        List of cluster center points as (x, y).
    """
    if not points:
        return []
    points_np = np.array(points)
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold
    ).fit(points_np)

    clustered = defaultdict(list)
    for label, pt in zip(clustering.labels_, points):
        clustered[label].append(pt)

    return [tuple(np.mean(pts, axis=0).astype(int)) for pts in clustered.values()]

def sort_into_grid(points, rows=9, cols=9):
    """
    Sorts clustered intersection points into a structured grid.

    Args:
        points (list): List of clustered intersection points.
        rows (int): Expected number of grid rows (default = 9).
        cols (int): Expected number of grid columns (default = 9).

    Returns:
        grid (list of lists): 2D array of grid points with shape rows×cols.
    """
    points = np.array(points)
    ys = np.array([pt[1] for pt in points])
    y_centers = np.sort(np.unique(np.round(ys / 50) * 50))

    grid = []
    for y in y_centers:
        row_points = [pt for pt in points if abs(pt[1] - y) < 30]
        row = sorted(row_points, key=lambda pt: pt[0])
        if len(row) == cols:
            grid.append(row[:cols])
    return grid

# -------------------- SECONDARY PROCESSING --------------------
def secondary_processing(warped, secondary_houghline, secondary_clustering):
    """
    Detects intersections of chessboard grid lines in a warped board image.

    Args:
        warped (np.ndarray): Warped chessboard image (top-down).
        secondary_houghline (int): Hough threshold (0 = predict via ML model).
        secondary_clustering (int): Clustering threshold (0 = predict via ML model).

    Returns:
        cluster_wrapped_pts (list): Clustered intersection points.
        img4 (np.ndarray): Visualization image with detected lines and intersections.
        secondary_hough_training_data (dict): Extracted Hough features.
        secondary_clustering_data (dict): Extracted clustering features.
    """
    # --- Edge Detection ---
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blurred_warped = cv2.GaussianBlur(warped_gray, (5, 5), 0)
    edges_warped = cv2.Canny(blurred_warped, 50, 150, apertureSize=3)

    # --- Feature Extraction ---
    secondary_hough_training_data["mean_intensity"] = np.mean(edges_warped)
    secondary_hough_training_data["std_intensity"] = np.std(edges_warped)
    secondary_hough_training_data["edge_density"] = np.sum(edges_warped > 0) / edges_warped.size

    sobel_x = cv2.Sobel(warped_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(warped_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    secondary_hough_training_data["gradient_mean"] = np.mean(gradient_magnitude)
    secondary_hough_training_data["gradient_std"] = np.std(gradient_magnitude)

    secondary_hough_training_data["entropy"] = shannon_entropy(warped_gray)
    secondary_hough_training_data["aspect_ratio"] = warped_gray.shape[1] / warped_gray.shape[0]

    # --- Predict Hough Threshold if needed ---
    if secondary_houghline == 0:
        secondary_hough_features = np.array([
            secondary_hough_training_data["mean_intensity"],
            secondary_hough_training_data["std_intensity"],
            secondary_hough_training_data["edge_density"],
            secondary_hough_training_data["gradient_mean"],
            secondary_hough_training_data["gradient_std"],
            secondary_hough_training_data["entropy"],
            secondary_hough_training_data["aspect_ratio"]
        ])
        secondary_hough_model = joblib.load('models/secondary_hough_pipeline.pkl')
        secondary_houghline = int(secondary_hough_model.predict([secondary_hough_features])[0])
        secondary_hough_training_data["hough_threshold"] = secondary_houghline
        print(f"Predicted Secondary Hough Threshold: {secondary_houghline}")

    # --- Detect Grid Lines with Hough Transform ---
    lines_warped = cv2.HoughLines(edges_warped, 1, np.pi / 180, secondary_houghline)
    img4 = warped.copy()
    if lines_warped is None:
        print("No lines found in warped image")
        exit()
    secondary_hough_training_data["hough_threshold"] = secondary_houghline

    # Draw detected lines
    for rho_theta in lines_warped:
        rho, theta = rho_theta[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
        x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
        cv2.line(img4, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("Secondary Detected Lines", img4)

    # --- Classify Lines into Vertical/Horizontal ---
    wrapped_vertical_lines, wrapped_horizontal_lines = [], []
    for rho_theta in lines_warped:
        rho, theta = rho_theta[0]
        angle = np.degrees(theta)
        if abs(angle) < 30 or abs(angle - 180) < 30:
            wrapped_vertical_lines.append((rho, theta))
        elif abs(angle - 90) < 30:
            wrapped_horizontal_lines.append((rho, theta))

    # --- Compute Intersections ---
    wrapped_intersections = []
    for vrho, vtheta in wrapped_vertical_lines:
        for hrho, htheta in wrapped_horizontal_lines:
            pt = compute_intersection(vrho, vtheta, hrho, htheta)
            if pt is not None:
                x, y = pt
                if 0 <= x < warped.shape[1] and 0 <= y < warped.shape[0]:
                    wrapped_intersections.append((x, y))

    # --- Clustering Features ---
    secondary_clustering_data["num_points"] = len(wrapped_intersections)
    if len(wrapped_intersections) >= 2:
        distances = pdist(np.array(wrapped_intersections))
        secondary_clustering_data["mean_distance"] = np.mean(distances)
        secondary_clustering_data["std_distance"] = np.std(distances)

    # --- Predict Clustering Threshold if needed ---
    if secondary_clustering == 0:
        secondary_clustering_features = np.array([
            secondary_clustering_data["num_points"],
            secondary_clustering_data["mean_distance"],
            secondary_clustering_data["std_distance"]
        ])
        secondary_clustering_model = joblib.load('models/secondary_cluster_pipeline.pkl')
        secondary_clustering = int(secondary_clustering_model.predict([secondary_clustering_features])[0])
        secondary_clustering_data["cluster_threshold"] = secondary_clustering
        print(f"Predicted Secondary Clustering Threshold: {secondary_clustering}")

    # --- Cluster Intersections ---
    cluster_wrapped_pts = cluster_points(wrapped_intersections, threshold=secondary_clustering)
    secondary_clustering_data["cluster_threshold"] = secondary_clustering

    # Visualize clustered points
    for pt in cluster_wrapped_pts:
        cv2.circle(img4, pt, 4, (255, 0, 0), -1)
    cv2.imshow("Warped Intersections", img4)

    return cluster_wrapped_pts, img4, secondary_hough_training_data, secondary_clustering_data

# -------------------- GRID VISUALIZATION --------------------
def visualize_grid(vis_squares, sorted_grid):
    """
    Draws the detected 8×8 grid of chessboard squares with labels.

    Args:
        vis_squares (np.ndarray): Copy of warped image to draw on.
        sorted_grid (list): 9x9 array of grid intersections.

    Returns:
        vis_squares (np.ndarray): Image annotated with square boundaries and labels.
    """
    for i in range(8):
        for j in range(8):
            # 4 corners of the square
            tl, tr = sorted_grid[i][j], sorted_grid[i][j+1]
            br, bl = sorted_grid[i+1][j+1], sorted_grid[i+1][j]

            # Draw square outline
            pts = np.array([tl, tr, br, bl], np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_squares, [pts], isClosed=True, color=(0, 255, 255), thickness=1)

            # Add square label (A–H, 1–8)
            cx, cy = int((tl[0] + br[0]) / 2), int((tl[1] + br[1]) / 2)
            label = f"{chr(ord('A') + i)}{8-j}"
            cv2.putText(vis_squares, label, (cx - 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    return vis_squares
