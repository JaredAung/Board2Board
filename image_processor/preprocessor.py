"""
Board2Board – Primary Preprocessing Module
Author: Jared Aung © 2025 All rights reserved.

Description:
------------
This module performs the **primary preprocessing stage** of chessboard detection:
1. Applies edge detection (Canny) and Hough Transform to detect chessboard lines.
2. Computes intersections of vertical and horizontal lines.
3. Clusters intersection points to remove duplicates and noise.
4. Identifies four board corners (top-left, top-right, bottom-right, bottom-left).
5. Warps the detected chessboard into a normalized square (top-down perspective).

Additionally:
- Extracts training features (intensity, edge density, entropy, etc.) 
  to log for regression models predicting optimal thresholds.
- Integrates ML models (loaded via `joblib`) to predict thresholds if thresholds are set to 0.
"""

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist
from collections import defaultdict
from skimage.measure import shannon_entropy
import joblib

# -------------------- DATA STRUCTURES --------------------
# Stores features collected during Hough line detection
primary_hough_training_data = {
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

# Stores features collected during intersection clustering
primary_clustering_data = {
    "num_points": 0,
    "mean_distance": 0.0,
    "std_distance": 0.0,
    "cluster_threshold": 0
}

# -------------------- HELPER FUNCTIONS --------------------
def compute_intersection(rho1, theta1, rho2, theta2):
    """
    Computes intersection point of two Hough lines (in polar form).

    Args:
        rho1, theta1: Parameters of the first line.
        rho2, theta2: Parameters of the second line.

    Returns:
        (x, y) tuple of intersection point (int), or None if lines are parallel.
    """
    a1, b1 = np.cos(theta1), np.sin(theta1)
    a2, b2 = np.cos(theta2), np.sin(theta2)
    A = np.array([[a1, b1], [a2, b2]])
    B = np.array([[rho1], [rho2]])
    if np.abs(np.linalg.det(A)) < 1e-10:  # Avoid nearly parallel lines
        return None
    intersection = np.linalg.solve(A, B)
    return tuple(np.round(intersection.flatten()).astype(int))

def cluster_points(points, threshold=40):
    """
    Clusters raw intersection points into averaged points using Agglomerative Clustering.

    Args:
        points (list): List of (x, y) points.
        threshold (int): Distance threshold for clustering.

    Returns:
        clustered_points (list): List of averaged cluster centers as (x, y).
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

def find_board_corner(points):
    """
    Identifies four board corners from clustered intersection points.

    Args:
        points (np.ndarray): Nx2 array of (x, y) points.

    Returns:
        src_points (np.ndarray): Four ordered corners (tl, tr, br, bl).
    """
    sums = points[:, 0] + points[:, 1]
    diffs = points[:, 0] - points[:, 1]

    tl = points[np.argmin(sums)]
    br = points[np.argmax(sums)]
    tr = points[np.argmin(diffs)]
    bl = points[np.argmax(diffs)]

    src_points = np.float32([tl, tr, br, bl])
    return src_points

# -------------------- MAIN PREPROCESSING --------------------
def preprocess_image(image, primary_houghline, primary_clustering):
    """
    Runs primary preprocessing to detect chessboard corners.

    Args:
        image (np.ndarray): Input chessboard image (BGR).
        primary_houghline (int): Threshold for Hough Transform (0 = predict with ML model).
        primary_clustering (int): Threshold for clustering intersections (0 = predict with ML model).

    Returns:
        src_points (np.ndarray): Four ordered chessboard corners (tl, tr, br, bl).
        primary_hough_training_data (dict): Extracted Hough training features.
        primary_clustering_data (dict): Extracted clustering training features.
    """
    # --- Preprocessing & Edge Detection ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # --- Feature Extraction ---
    primary_hough_training_data["mean_intensity"] = np.mean(edges)
    primary_hough_training_data["std_intensity"] = np.std(edges)
    primary_hough_training_data["edge_density"] = np.sum(edges > 0) / edges.size

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    primary_hough_training_data["gradient_mean"] = np.mean(gradient_magnitude)
    primary_hough_training_data["gradient_std"] = np.std(gradient_magnitude)

    primary_hough_training_data["entropy"] = shannon_entropy(gray)
    primary_hough_training_data["aspect_ratio"] = gray.shape[1] / gray.shape[0]

    # --- Predict Hough Threshold if not provided ---
    if primary_houghline == 0:
        primary_hough_features = np.array([
            primary_hough_training_data["mean_intensity"],
            primary_hough_training_data["std_intensity"],
            primary_hough_training_data["edge_density"],
            primary_hough_training_data["gradient_mean"],
            primary_hough_training_data["gradient_std"],
            primary_hough_training_data["entropy"],
            primary_hough_training_data["aspect_ratio"]
        ])
        primary_hough_model = joblib.load('models/primary_hough_pipeline.pkl')
        primary_houghline = int(primary_hough_model.predict([primary_hough_features])[0])
        print(f"Predicted Primary Hough Threshold: {primary_houghline}")

    # --- Detect Lines with Hough Transform ---
    lines = cv2.HoughLines(edges, 1, np.pi / 180, primary_houghline)
    if lines is None:
        print("No lines found")
        exit()
    primary_hough_training_data["hough_threshold"] = primary_houghline

    # Draw detected lines for visualization
    img2 = image.copy()
    for rho_theta in lines:
        rho, theta = rho_theta[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
        x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
        cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("Primary Hough Lines", img2)

    # --- Classify Lines into Vertical/Horizontal ---
    vertical_lines, horizontal_lines = [], []
    for rho_theta in lines:
        rho, theta = rho_theta[0]
        angle = np.degrees(theta)
        if abs(angle) < 30 or abs(angle - 180) < 30:
            vertical_lines.append((rho, theta))
        elif abs(angle - 90) < 30:
            horizontal_lines.append((rho, theta))

    # --- Find Intersections of Vertical & Horizontal Lines ---
    intersections = []
    img3 = image.copy()
    for vrho, vtheta in vertical_lines:
        for hrho, htheta in horizontal_lines:
            pt = compute_intersection(vrho, vtheta, hrho, htheta)
            if pt is not None:
                x, y = pt
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    intersections.append((x, y))

    # --- Compute Clustering Features ---
    primary_clustering_data["cluster_threshold"] = primary_clustering
    points = np.array(intersections)
    primary_clustering_data["num_points"] = len(points)

    if len(points) >= 2:
        distances = pdist(points)
        primary_clustering_data["mean_distance"] = np.mean(distances)
        primary_clustering_data["std_distance"] = np.std(distances)
    else:
        primary_clustering_data["mean_distance"] = 0
        primary_clustering_data["std_distance"] = 0

    # --- Predict Clustering Threshold if not provided ---
    if primary_clustering == 0:
        primary_clustering_features = np.array([
            primary_clustering_data["num_points"],
            primary_clustering_data["mean_distance"],
            primary_clustering_data["std_distance"]
        ])
        primary_clustering_model = joblib.load('models/primary_cluster_pipeline.pkl')
        primary_clustering = int(primary_clustering_model.predict([primary_clustering_features])[0])
        print(f"Predicted Primary Clustering Threshold: {primary_clustering}")
        primary_clustering_data["cluster_threshold"] = primary_clustering

    # --- Cluster Intersections & Find Board Corners ---
    clustered_pts = cluster_points(intersections, threshold=primary_clustering)
    pts = np.array(clustered_pts)
    src_points = find_board_corner(pts)

    # Visualize detected corners
    for corner in src_points:
        cv2.circle(img3, tuple(map(int, corner)), 5, (0, 0, 255), -1)
    cv2.imshow("Board Corners", img3)

    return src_points, primary_hough_training_data, primary_clustering_data

# -------------------- WARPING FUNCTION --------------------
def warp_board_by_corners(src_points, image):
    """
    Warps the chessboard image into a normalized square (top-down view).

    Args:
        src_points (np.ndarray): Four detected board corners.
        image (np.ndarray): Original chessboard image.

    Returns:
        warped (np.ndarray): Warped 800x800 chessboard image.
    """
    board_size = 800
    dst_points = np.float32([
        [10, 10],
        [board_size - 10, 10],
        [board_size - 10, board_size - 10],
        [10, board_size - 10]
    ])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, M, (board_size, board_size))
    cv2.imshow("Warped Board", warped)
    return warped
