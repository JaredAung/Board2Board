import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import os
from skimage.measure import shannon_entropy
from scipy.spatial.distance import pdist
from image_processor import preprocessor,secondary_processor

param_states = [
    {"name": "hough_threshold", "value": 150},  # Initial threshold for Hough Transform
    {"name": "cluster_threshold", "value": 70},  # Initial threshold for clustering points
    {"name": "primary_hough_threshold", "value": 170},  # Threshold for primary Hough lines
    {"name": "primary_cluster_threshold", "value": 85},  # Threshold for primary clustering
]

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

primary_clustering_data = {
        "num_points": 0,
        "mean_distance": 0.0,
        "std_distance": 0.0,
        "cluster_threshold": 0
    }

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

secondary_clustering_data = {
        "num_points": 0,
        "mean_distance": 0.0,
        "std_distance": 0.0,
        "cluster_threshold": 0
    }

def save_extracted_images(sorted_grid, warped,dir):

    for i in range(8):
        for j in range(8):
            # 4 corners of the square
            tl = sorted_grid[i][j]
            tr = sorted_grid[i][j+1]
            br = sorted_grid[i+1][j+1]
            bl = sorted_grid[i+1][j]

            # Perspective crop: map square to 224x224 output
            src_pts = np.float32([tl, tr, br, bl])
            dst_pts = np.float32([[0, 0], [224, 0], [224, 224], [0, 224]])

            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            square_crop = cv2.warpPerspective(warped, M, (224, 224))
            square_crop = cv2.convertScaleAbs(square_crop,alpha=1.1,beta=10)
            file = f"{chr(ord('A') + i)}{8-j}"

            cv2.imwrite(f"{dir}/{file}.jpg", square_crop)

def save_threshold_data():
    with open('primary_hough_training.txt', 'a') as f:
        print(f"Writing to primary_hough_training.txt: {primary_hough_training_data}")
        f.write(f"{primary_hough_training_data['mean_intensity']}, {primary_hough_training_data['std_intensity']}, {primary_hough_training_data['edge_density']}, {primary_hough_training_data['gradient_mean']}, {primary_hough_training_data['gradient_std']}, {primary_hough_training_data['entropy']}, {primary_hough_training_data['aspect_ratio']}, {primary_hough_training_data['hough_threshold']}\n")
    with open('primary_clustering_training.txt', 'a') as f:
        print(f"Writing to primary_clustering_training.txt: {primary_clustering_data}")
        f.write(f"{primary_clustering_data['num_points']}, {primary_clustering_data['mean_distance']}, {primary_clustering_data['std_distance']}, {primary_clustering_data['cluster_threshold']}\n")
    with open('secondary_hough_training.txt', 'a') as f:
        print(f"Writing to secondary_hough_training.txt: {secondary_hough_training_data}")
        f.write(f"{secondary_hough_training_data['mean_intensity']}, {secondary_hough_training_data['std_intensity']}, {secondary_hough_training_data['edge_density']}, {secondary_hough_training_data['gradient_mean']}, {secondary_hough_training_data['gradient_std']}, {secondary_hough_training_data['entropy']}, {secondary_hough_training_data['aspect_ratio']}, {secondary_hough_training_data['hough_threshold']}\n")
    with open('secondary_clustering_training.txt', 'a') as f:
        print(f"Writing to secondary_clustering_training.txt: {secondary_clustering_data}")
        f.write(f"{secondary_clustering_data['num_points']}, {secondary_clustering_data['mean_distance']}, {secondary_clustering_data['std_distance']}, {secondary_clustering_data['cluster_threshold']}\n")

if __name__ == "__main__":
    image = cv2.imread('testing-images/15.jpg') # board image path
    exit = 0
    exit1 = 0
    exit2 = 0

    while exit == 0:

        while exit1 == 0:
            src_points, primary_hough_training_data, primary_clustering_data = preprocessor.preprocess_image(image,param_states[2]["value"], param_states[3]["value"])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            decision = input("Is the board orientation correct? (y/n): ").strip().lower()
            if decision == 'y':
                cv2.destroyAllWindows()
                break
            cv2.destroyAllWindows()
            param_states[2]["value"] = int(input(f"Enter Hough threshold (current: {param_states[2]['value']}): ") or param_states[2]["value"])
            param_states[3]["value"] = int(input(f"Enter clustering threshold (current: {param_states[3]['value']}): ") or param_states[3]["value"])

        warped = preprocessor.warp_board_by_corners(src_points, image)

        # Second processing
        cluster_wrapped_pts, img4, secondary_hough_training_data, secondary_clustering_data = secondary_processor.secondary_processing(warped, param_states[0]["value"], param_states[1]["value"])
        sorted_grid = secondary_processor.sort_into_grid(cluster_wrapped_pts, rows=9, cols=9)
        while exit2 == 0:
            print("Sorted Grid Size:", len(sorted_grid), "x", len(sorted_grid[0]) if sorted_grid else 0)
            cv2.imshow(f"Sorted Grid {len(sorted_grid)}x{len(sorted_grid[0]) if sorted_grid else 0}", img4)
            cv2.waitKey(0)
            decision = input("Does the grid look correct? (y/n): ").strip().lower()
            if decision == 'y':
                cv2.destroyAllWindows()
                break
            param_states[0]["value"] = int(input(f"Enter Hough threshold for secondary processing (current: {param_states[0]['value']}): ") or param_states[0]["value"])
            param_states[1]["value"] = int(input(f"Enter clustering threshold for grid sorting (current: {param_states[1]['value']}): ") or param_states[1]["value"])
            cv2.destroyAllWindows()
            cluster_wrapped_pts, img4, secondary_hough_training_data, secondary_clustering_data = secondary_processor.secondary_processing(warped, param_states[0]["value"], param_states[1]["value"])
            sorted_grid = secondary_processor.sort_into_grid(cluster_wrapped_pts, rows=9, cols=9)

        print("Final Sorted Grid Size:", len(sorted_grid), "x", len(sorted_grid[0]) if sorted_grid else 0)
        vis_squares = warped.copy()

        vis_squares = secondary_processor.visualize_grid(vis_squares, sorted_grid)
        cv2.imshow("Squares Visualization", vis_squares)
        cv2.waitKey(0)

        decision = input("Does the grid look correct? (y/n): " ).strip().lower()
        if decision == 'y':
            exit1 = 1
        cv2.destroyAllWindows()
        
        lastDecision = input("Is this final processing correct? (y/n): ").strip().lower()
        if lastDecision == 'y':
            exit = 1

    save_threshold_data()
    save_extracted_images(sorted_grid, warped, dir = "testing-images/extracted")


    print("Processing complete. Images saved.")
    cv2.destroyAllWindows()
    
