"""
Board2Board – Chessboard Preprocessing & Data Extraction
Author: Jared Aung © 2025 All rights reserved.

Description:
------------
This script is desgined to generate training data and extract chessboard squares
from an input image of a chessboard. It processes a chessboard image through 
two main stages:
1. Primary preprocessing – detects board orientation using Hough Transform 
   and clustering, then warps the board into a normalized top-down view.
2. Secondary preprocessing – detects and sorts the 9×9 grid of intersections.

Finally save the extracted squares in a folder (manually labelling required) and
threshold data with image features in a csv file for training.

Outputs:
- Extracted square crops (224×224 px) for each of the 64 chessboard squares.
- Threshold data from Hough and clustering stages (logged for training).
- Visualization of the processed grid for validation.

This script is designed for interactive use:
- The user validates orientation and grid correctness, and can adjust thresholds
  as needed until satisfactory results are achieved.
"""

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import os
from skimage.measure import shannon_entropy
from scipy.spatial.distance import pdist
from image_processor import preprocessor, secondary_processor

# -------------------- PARAMETER STATES --------------------
# Stores configurable thresholds for Hough Transform and clustering.
param_states = [
    {"name": "hough_threshold", "value": 150},   # Secondary Hough Transform
    {"name": "cluster_threshold", "value": 70},  # Secondary clustering
    {"name": "primary_hough_threshold", "value": 170},  # Primary Hough Transform
    {"name": "primary_cluster_threshold", "value": 85}, # Primary clustering
]

# Training data collectors for primary Hough step
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

# Training data collectors for primary clustering step
primary_clustering_data = {
    "num_points": 0,
    "mean_distance": 0.0,
    "std_distance": 0.0,
    "cluster_threshold": 0
}

# Training data collectors for secondary Hough step
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

# Training data collectors for secondary clustering step
secondary_clustering_data = {
    "num_points": 0,
    "mean_distance": 0.0,
    "std_distance": 0.0,
    "cluster_threshold": 0
}

# -------------------- IMAGE EXTRACTION --------------------
def save_extracted_images(sorted_grid, warped, dir):
    """
    Saves 64 extracted chessboard squares as images (224x224).

    Args:
        sorted_grid (list): 9x9 grid of intersection points (corners).
        warped (np.ndarray): Warped top-down chessboard image.
        dir (str): Output directory path to save square images.

    Each square is perspective-transformed into a normalized 224x224 patch.
    Files are named using chess notation (A1–H8).
    """
    for i in range(8):
        for j in range(8):
            # Extract the four corners of the square
            tl = sorted_grid[i][j]
            tr = sorted_grid[i][j+1]
            br = sorted_grid[i+1][j+1]
            bl = sorted_grid[i+1][j]

            # Define source and destination points for perspective transform
            src_pts = np.float32([tl, tr, br, bl])
            dst_pts = np.float32([[0, 0], [224, 0], [224, 224], [0, 224]])

            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            square_crop = cv2.warpPerspective(warped, M, (224, 224))

            # Apply simple brightness/contrast adjustment
            square_crop = cv2.convertScaleAbs(square_crop, alpha=1.1, beta=10)

            # Save using chess notation (A–H, 1–8)
            file = f"{chr(ord('A') + i)}{8-j}"
            cv2.imwrite(f"{dir}/{file}.jpg", square_crop)

# -------------------- THRESHOLD DATA LOGGING --------------------
def save_threshold_data():
    """
    Appends Hough and clustering threshold training data 
    to log files for later analysis/modeling.
    """
    with open('primary_hough_training.txt', 'a') as f:
        print(f"Writing to primary_hough_training.txt: {primary_hough_training_data}")
        f.write(f"{primary_hough_training_data['mean_intensity']}, "
                f"{primary_hough_training_data['std_intensity']}, "
                f"{primary_hough_training_data['edge_density']}, "
                f"{primary_hough_training_data['gradient_mean']}, "
                f"{primary_hough_training_data['gradient_std']}, "
                f"{primary_hough_training_data['entropy']}, "
                f"{primary_hough_training_data['aspect_ratio']}, "
                f"{primary_hough_training_data['hough_threshold']}\n")

    with open('primary_clustering_training.txt', 'a') as f:
        print(f"Writing to primary_clustering_training.txt: {primary_clustering_data}")
        f.write(f"{primary_clustering_data['num_points']}, "
                f"{primary_clustering_data['mean_distance']}, "
                f"{primary_clustering_data['std_distance']}, "
                f"{primary_clustering_data['cluster_threshold']}\n")

    with open('secondary_hough_training.txt', 'a') as f:
        print(f"Writing to secondary_hough_training.txt: {secondary_hough_training_data}")
        f.write(f"{secondary_hough_training_data['mean_intensity']}, "
                f"{secondary_hough_training_data['std_intensity']}, "
                f"{secondary_hough_training_data['edge_density']}, "
                f"{secondary_hough_training_data['gradient_mean']}, "
                f"{secondary_hough_training_data['gradient_std']}, "
                f"{secondary_hough_training_data['entropy']}, "
                f"{secondary_hough_training_data['aspect_ratio']}, "
                f"{secondary_hough_training_data['hough_threshold']}\n")

    with open('secondary_clustering_training.txt', 'a') as f:
        print(f"Writing to secondary_clustering_training.txt: {secondary_clustering_data}")
        f.write(f"{secondary_clustering_data['num_points']}, "
                f"{secondary_clustering_data['mean_distance']}, "
                f"{secondary_clustering_data['std_distance']}, "
                f"{secondary_clustering_data['cluster_threshold']}\n")

# -------------------- MAIN EXECUTION --------------------
if __name__ == "__main__":
    # Load test image
    image = cv2.imread('testing-images/15.jpg')
    exit, exit1, exit2 = 0, 0, 0

    # -------------------- PRIMARY PROCESSING LOOP --------------------
    """
    Step 1: Primary preprocessing
    - Detects board orientation with Hough Transform & clustering.
    - Interactive threshold tuning until orientation is correct.
    """
    while exit == 0:
        while exit1 == 0:
            src_points, primary_hough_training_data, primary_clustering_data = preprocessor.preprocess_image(
                image,
                param_states[2]["value"],
                param_states[3]["value"]
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # User confirms or re-tunes thresholds
            decision = input("Is the board orientation correct? (y/n): ").strip().lower()
            if decision == 'y':
                break
            param_states[2]["value"] = int(input(
                f"Enter Hough threshold (current: {param_states[2]['value']}): "
            ) or param_states[2]["value"])
            param_states[3]["value"] = int(input(
                f"Enter clustering threshold (current: {param_states[3]['value']}): "
            ) or param_states[3]["value"])

        # Warp chessboard into top-down square
        warped = preprocessor.warp_board_by_corners(src_points, image)

        # -------------------- SECONDARY PROCESSING LOOP --------------------
        """
        Step 2: Secondary preprocessing
        - Detects and sorts 9x9 grid intersections on warped image.
        - Interactive threshold tuning until grid looks correct.
        """
        cluster_wrapped_pts, img4, secondary_hough_training_data, secondary_clustering_data = secondary_processor.secondary_processing(
            warped,
            param_states[0]["value"],
            param_states[1]["value"]
        )
        sorted_grid = secondary_processor.sort_into_grid(cluster_wrapped_pts, rows=9, cols=9)

        while exit2 == 0:
            print("Sorted Grid Size:", len(sorted_grid), "x", len(sorted_grid[0]) if sorted_grid else 0)
            cv2.imshow(f"Sorted Grid {len(sorted_grid)}x{len(sorted_grid[0]) if sorted_grid else 0}", img4)
            cv2.waitKey(0)

            decision = input("Does the grid look correct? (y/n): ").strip().lower()
            if decision == 'y':
                break

            # Allow manual threshold tuning
            param_states[0]["value"] = int(input(
                f"Enter Hough threshold for secondary processing (current: {param_states[0]['value']}): "
            ) or param_states[0]["value"])
            param_states[1]["value"] = int(input(
                f"Enter clustering threshold for grid sorting (current: {param_states[1]['value']}): "
            ) or param_states[1]["value"])

            # Re-run with updated thresholds
            cluster_wrapped_pts, img4, secondary_hough_training_data, secondary_clustering_data = secondary_processor.secondary_processing(
                warped,
                param_states[0]["value"],
                param_states[1]["value"]
            )
            sorted_grid = secondary_processor.sort_into_grid(cluster_wrapped_pts, rows=9, cols=9)

        # Final visualization of grid
        print("Final Sorted Grid Size:", len(sorted_grid), "x", len(sorted_grid[0]) if sorted_grid else 0)
        vis_squares = secondary_processor.visualize_grid(warped.copy(), sorted_grid)
        cv2.imshow("Squares Visualization", vis_squares)
        cv2.waitKey(0)

        if input("Does the grid look correct? (y/n): ").strip().lower() == 'y':
            exit1 = 1
        if input("Is this final processing correct? (y/n): ").strip().lower() == 'y':
            exit = 1

    # -------------------- SAVE OUTPUTS --------------------
    save_threshold_data()
    save_extracted_images(sorted_grid, warped, dir="testing-images/extracted")

    print("Processing complete. Images saved.")
    cv2.destroyAllWindows()
