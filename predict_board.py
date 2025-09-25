import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import os
from skimage.measure import shannon_entropy
from scipy.spatial.distance import pdist
import joblib
from keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from image_processor import preprocessor, secondary_processor
import chess
import chess.svg

param_states = [
    {"name": "hough_threshold", "value": 0},  # Initial threshold for Hough Transform
    {"name": "cluster_threshold", "value": 0},  # Initial threshold for clustering points
    {"name": "primary_hough_threshold", "value": 0},  # Threshold for primary Hough lines
    {"name": "primary_cluster_threshold", "value": 0},  # Threshold for primary clustering
]
primary_hough_data = {
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

def show_class(class_index):
    match class_index:
        case 0:
            return "b"
        case 1:
            return "k"
        case 2:
            return "n"
        case 3:
            return "p"
        case 4:
            return "q"
        case 5:
            return "r"
        case 6:
            return "1"
        case 7:
            return "B"
        case 8:
            return "K"
        case 9:
            return "N"
        case 10:
            return "P"
        case 11:
            return "Q"
        case 12:
            return "R"

grid = [[0 for _ in range(8)] for _ in range(8)]
def predict_and_build(sorted_grid, warped):
    #fen = ""
    ones = 0
    piece_model = load_model('models/board2board.h5')
    for j in range(8):
        for i in range(8):
            tl = sorted_grid[i][j]
            tr = sorted_grid[i][j+1]
            br = sorted_grid[i+1][j+1]
            bl = sorted_grid[i+1][j]

            square = warped[tl[1]:br[1], tl[0]:br[0]]
            square = cv2.resize(square, (224, 224))
            square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
            square = preprocess_input(square)
            square = np.expand_dims(square, axis=0)
            predict = piece_model.predict(square)
            predict_index = np.argmax(predict)
            predict_class = show_class(predict_index)
            grid[j][i] = predict_class
            #confidence = np.max(predict)
            #result += f"{chr(ord('A') + i)}{8-j}: {predict_class}, {confidence}\n"
            # if predict_class == "1":
            #     ones += 1
            # else:
            #     fen += f"{ones}" if ones > 0 else ""
            #     ones = 0
            #     fen += predict_class

    #     if ones > 0:
    #         fen += f"{ones}"
    #         ones = 0
    #     fen += "/"
    # fen = fen[:-1] + " w KQkq - 0 1"
    print("Grid:")
    for row in grid:
        print(row)
    return grid

def visualize_predictions(warped, sorted_grid, grid):
    vis_predicted = warped.copy()
    for i in range(8):
        for j in range(8):
            tl = sorted_grid[i][j]
            tr = sorted_grid[i][j+1]
            br = sorted_grid[i+1][j+1]
            bl = sorted_grid[i+1][j]

            cx = int((tl[0] + br[0]) / 2)
            cy = int((tl[1] + br[1]) / 2)
            label = f"{f"{chr(ord('A') + i)}{8-j}"} = {grid[j][i]}"
            cv2.putText(vis_predicted, label, (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 255), 2)

    cv2.imshow("Predicted Pieces", vis_predicted)
    cv2.waitKey(0)

def build_fen(grid):
    fen = ""
    ones = 0
    for j in range(8):
        for i in range(8):

            if grid[j][i] == "1":
                ones += 1
            else:
                fen += f"{ones}" if ones > 0 else ""
                ones = 0
                fen += grid[j][i]

        if ones > 0:
            fen += f"{ones}"
            ones = 0
        fen += "/"
    fen = fen[:-1] + " w KQkq - 0 1"
    return fen

if __name__ == "__main__":
    filename = '15'
    image = cv2.imread(f'testing-images/{filename}.jpg') # board image path
    exit = 0
    exit1 = 0
    exit2 = 0

    while exit == 0:

        while exit1 == 0:
            src_points, primary_hough_data, primary_clustering_data = preprocessor.preprocess_image(image,param_states[2]["value"],param_states[3]["value"])
            param_states[2]["value"] = primary_hough_data["hough_threshold"]
            param_states[3]["value"] = primary_clustering_data["cluster_threshold"]
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            decision = input("Is the board orientation correct? (y/n): ").strip().lower()
            if decision == 'y':
                cv2.destroyAllWindows()
                break
            cv2.destroyAllWindows()
            param_states[2]["value"] = int(input(f"Enter Hough threshold (current: {param_states[2]['value']}): ") or param_states[2]["value"])
            param_states[3]["value"] = int(input(f"Enter clustering threshold (current: {param_states[3]['value']}): ") or param_states[3]["value"])
        

        # board_size = 800
        # dst_points = np.float32([[10, 10], [board_size - 10, 10], [board_size - 10, board_size - 10], [10, board_size - 10]])
        # M = cv2.getPerspectiveTransform(src_points, dst_points)
        # warped = cv2.warpPerspective(image, M, (board_size, board_size)) 
        # cv2.imshow("Warped Board", warped)

        warped = preprocessor.warp_board_by_corners(src_points, image)


        # Second processing
        cluster_wrapped_pts, img4, secondary_hough_training_data, secondary_clustering_data = secondary_processor.secondary_processing(warped,param_states[0]["value"],param_states[1]["value"])
        param_states[0]["value"] = secondary_hough_training_data["hough_threshold"]
        param_states[1]["value"] = secondary_clustering_data["cluster_threshold"]
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
            cluster_wrapped_pts, img4, secondary_hough_training_data, secondary_clustering_data = secondary_processor.secondary_processing(warped,param_states[0]["value"],param_states[1]["value"])
            
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


    grid = predict_and_build(sorted_grid, warped)
    visualize_predictions(warped, sorted_grid, grid)

    while True:
        final_approval = input("Would you like to make any piece corrections? (y/n): ").strip().lower()
        if final_approval == 'n':
            break

        position = input("Enter position to correct (e.g., E2): ").strip().upper()
        if len(position) != 2 or position[0] not in "ABCDEFGH" or position[1] not in "12345678":
            print("Invalid position format. Please use format like E2.")
            continue

        piece = input("Enter correct piece (e.g., P, n, 1 for empty): ").strip()
        if piece not in ["b", "k", "n", "p", "q", "r", "1", "B", "K", "N", "P", "Q", "R"]:
            print("Invalid piece. Use standard notation (e.g., P, n, 1 for empty).")
            continue

        grid[8 - int(position[1])][ord(position[0]) - ord('A')] = piece
        visualize_predictions(warped, sorted_grid, grid)

    
    fen = build_fen(grid)
    print(f"FEN: {fen}")

    final_board = chess.Board(fen)
    print(final_board)
    cv2.destroyAllWindows()

    chess_svg = chess.svg.board(final_board)
    with open(f"predicted/predicted_board_{filename}.svg", "w") as f:
        f.write(chess_svg)
        print(f"SVG file saved as predicted/predicted_board_{filename}.svg")

## © 2025 All rights reserved. Jared Aung
