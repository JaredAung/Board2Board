"""
Board2Board – Chessboard Recognition Pipeline
Author: Jared Aung © 2025 All rights reserved.

File: predict_board.py
Description:
------------
This script processes an over-the-board chess image and converts it into a 
Forsyth-Edwards Notation (FEN) string. The pipeline consists of:
1. Detecting board edges & corners (Hough Transform + clustering).
2. Warping the image to get a clean, top-down chessboard.
3. Detecting and sorting the 9x9 intersection grid.
4. Predicting pieces in each square using a trained CNN (ResNet-based).
5. Constructing a FEN string for the board state.
6. Allowing manual corrections of predictions.
7. Saving the final predicted board as an SVG.

Dependencies:
-------------
- OpenCV (cv2)
- NumPy
- scikit-learn (AgglomerativeClustering)
- scikit-image (shannon_entropy)
- SciPy (pdist)
- Keras / TensorFlow
- python-chess (board + SVG export)
- Custom modules: `image_processor.preprocessor`, `image_processor.secondary_processor`
"""

import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from image_processor import preprocessor, secondary_processor
import chess
import chess.svg

# -------------------- PARAMETER STATES --------------------
# Stores adjustable thresholds for Hough Transform & clustering.
param_states = [
    {"name": "hough_threshold", "value": 0},             # Secondary Hough Transform
    {"name": "cluster_threshold", "value": 0},           # Secondary clustering
    {"name": "primary_hough_threshold", "value": 0},     # Primary Hough Transform
    {"name": "primary_cluster_threshold", "value": 0},   # Primary clustering
]

# Metrics/features collected during primary HoughLine processing.
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

# Metrics for clustering step in primary processing.
primary_clustering_data = {
        "num_points": 0,
        "mean_distance": 0.0,
        "std_distance": 0.0,
        "cluster_threshold": 0
    }

# Metrics/features collected during secondary Hough processing.
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

# Metrics for clustering step in secondary processing.
secondary_clustering_data = {
        "num_points": 0,
        "mean_distance": 0.0,
        "std_distance": 0.0,
        "cluster_threshold": 0
    }

# Initialize an empty 8×8 grid for board representation.
grid = [[0 for _ in range(8)] for _ in range(8)]

# -------------------- CLASS LABEL MAPPING --------------------
def show_class(class_index):
    """
    Maps CNN class indices (0–12) to chess piece FEN notation.

    Lowercase = black pieces
    Uppercase = white pieces
    "1" = empty square
    """
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

# -------------------- PIECE PREDICTION --------------------
def predict_and_build(sorted_grid, warped):
    """
    Predicts the contents of each chessboard square using the trained CNN model.

    Args:
        sorted_grid: 9x9 array of intersection points (corners of squares).
        warped: top-down warped chessboard image.

    Returns:
        grid: 8x8 matrix filled with predicted piece symbols.
    """
    
    piece_model = load_model('models/board2board.h5')
    for j in range(8):
        for i in range(8):
            # Extract square by bounding box defined by 4 corners
            tl = sorted_grid[i][j]
            br = sorted_grid[i+1][j+1]
            square = warped[tl[1]:br[1], tl[0]:br[0]]

            # Preprocess square for ResNet input
            square = cv2.resize(square, (224, 224))
            square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
            square = preprocess_input(square)
            square = np.expand_dims(square, axis=0)

            # Predict piece in square
            predict = piece_model.predict(square)
            predict_index = np.argmax(predict)
            predict_class = show_class(predict_index)
            grid[j][i] = predict_class
            
    print("Grid:")
    for row in grid:
        print(row)
    return grid

# -------------------- VISUALIZATION --------------------
def visualize_predictions(warped, sorted_grid, grid):
    """
    Overlays predicted piece labels onto the warped chessboard image.

    Args:
        warped: warped chessboard image.
        sorted_grid: 9x9 array of grid intersections.
        grid: 8x8 predicted board matrix.
    """
    vis_predicted = warped.copy()
    for i in range(8):
        for j in range(8):
            tl = sorted_grid[i][j]
            br = sorted_grid[i+1][j+1]

            # Compute center of the square for label placement
            cx = int((tl[0] + br[0]) / 2)
            cy = int((tl[1] + br[1]) / 2)
            label = f"{f"{chr(ord('A') + i)}{8-j}"} = {grid[j][i]}"
            cv2.putText(vis_predicted, label, (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 255), 2)

    cv2.imshow("Predicted Pieces", vis_predicted)
    cv2.waitKey(0)

# -------------------- FEN CONSTRUCTION --------------------
def build_fen(grid) -> str:
    """
    Converts the predicted grid into a valid Forsyth-Edwards Notation (FEN) string.

    Args:
        grid: 8x8 board state matrix.

    Returns:
        fen: FEN string representation of the board.
    """
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

# -------------------- MAIN SCRIPT --------------------
if __name__ == "__main__":
    filename = '16' # File name 
    image = cv2.imread(f'testing-images/{filename}.jpg') # board image path
    exit = 0
    exit1 = 0
    exit2 = 0

    # Interactive threshold tuning loop (primary + secondary processing)
    # - Detect board orientation
    # - Adjust Hough/clustering thresholds until grid looks correct
    # - Warp the board for clean processing
    # - Finalize sorted grid of intersection points
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

    # Predict board state 
    grid = predict_and_build(sorted_grid, warped)
    visualize_predictions(warped, sorted_grid, grid)

    # Manual correction
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

    # Convert grid to FEN
    fen = build_fen(grid)
    print(f"FEN: {fen}")

    # Convert FEN to python-chess Board object
    final_board = chess.Board(fen)
    print(final_board)
    cv2.destroyAllWindows()

    # Save final board as SVG
    chess_svg = chess.svg.board(final_board)
    with open(f"predicted/predicted_board_{filename}.svg", "w") as f:
        f.write(chess_svg)
        print(f"SVG file saved as predicted/predicted_board_{filename}.svg")

    print("Operation complete.")