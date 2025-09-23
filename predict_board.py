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

param_states = [
    {"name": "hough_threshold", "value": 0},  # Initial threshold for Hough Transform
    {"name": "cluster_threshold", "value": 0},  # Initial threshold for clustering points
    {"name": "primary_hough_threshold", "value": 0},  # Threshold for primary Hough lines
    {"name": "primary_cluster_threshold", "value": 0},  # Threshold for primary clustering
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


def compute_intersection(rho1, theta1, rho2, theta2):
    a1, b1 = np.cos(theta1), np.sin(theta1)
    a2, b2 = np.cos(theta2), np.sin(theta2)
    A = np.array([[a1, b1], [a2, b2]])
    B = np.array([[rho1], [rho2]])
    if np.abs(np.linalg.det(A)) < 1e-10:
        return None
    intersection = np.linalg.solve(A, B)
    return tuple(np.round(intersection.flatten()).astype(int))

def cluster_points(points, threshold=40):
    if not points:
        return []
    points_np = np.array(points)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold).fit(points_np)
    clustered = defaultdict(list)
    for label, pt in zip(clustering.labels_, points):
        clustered[label].append(pt)
    return [tuple(np.mean(pts, axis=0).astype(int)) for pts in clustered.values()]      

def find_board_corner(points):
    sums = points[:, 0] + points[:, 1]
    diffs = points[:, 0] - points[:, 1]

    tl = points[np.argmin(sums)]
    br = points[np.argmax(sums)]
    tr = points[np.argmin(diffs)]
    bl = points[np.argmax(diffs)]

    src_points = np.float32([tl, tr, br, bl])
    return src_points

def sort_into_grid(points,rows=9,cols =9):
    points = np.array(points)
    ys = np.array([pt[1] for pt in points])
    y_centers = np.sort(np.unique(np.round(ys / 50) * 50))

    grid = []
    for y in y_centers:
        row_points = [pt for pt in points if abs(pt[1] - y) < 30]
        row = sorted(row_points, key=lambda pt: pt[0])
        if len(row) == 9:
            grid.append(row[:cols])
    return grid

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1) # could be 1 
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Mean and std intensity
    if (param_states[2]["value"] == 0):
        primary_hough_training_data["mean_intensity"] = np.mean(edges)
        primary_hough_training_data["std_intensity"] = np.std(edges)

        #Edge density
        primary_hough_training_data["edge_density"] = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        #Gradient magnitue
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        primary_hough_training_data["gradient_mean"] = np.mean(gradient_magnitude)
        primary_hough_training_data["gradient_std"] = np.std(gradient_magnitude)

        #Entropy
        primary_hough_training_data["entropy"] = shannon_entropy(gray)

        #Aspect_ratio
        primary_hough_training_data["aspect_ratio"] = gray.shape[1] / gray.shape[0]

        #cv2.imshow('Primary Edges', edges) # see edges
        primary_hough_features = np.array([primary_hough_training_data["mean_intensity"],
                                        primary_hough_training_data["std_intensity"],
                                        primary_hough_training_data["edge_density"],
                                        primary_hough_training_data["gradient_mean"],
                                        primary_hough_training_data["gradient_std"],
                                        primary_hough_training_data["entropy"],
                                        primary_hough_training_data["aspect_ratio"]])

        primary_hough_model = joblib.load('models/primary_hough_pipeline.pkl')
        param_states[2]["value"] = int(primary_hough_model.predict([primary_hough_features])[0])
        print(f"Predicted Primary Hough Threshold: {param_states[2]["value"]}")
    
    lines = cv2.HoughLines(edges, 1, np.pi / 180, param_states[2]["value"]) # 100 - 150 
    
    img2 = image.copy()
    for rho_theta in lines:
        rho, theta = rho_theta[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("Primary Hough Lines", img2)

    vertical_lines = []
    horizontal_lines = []
    
    for rho_theta in lines:
        rho, theta = rho_theta[0]
        angle = np.degrees(theta)
        if abs(angle) < 30 or abs(angle - 180) < 30: #maybe 40
            vertical_lines.append((rho, theta))
        elif abs(angle - 90) < 30:
            horizontal_lines.append((rho, theta))

    intersections = []
    img3 = image.copy()

    for vrho, vtheta in vertical_lines:
        for hrho, htheta in horizontal_lines:
            pt = compute_intersection(vrho, vtheta, hrho, htheta)
            if pt is not None:
                x, y = pt
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    intersections.append((x, y))
                    

    
    if (param_states[3]["value"] == 0):
        points = np.array(intersections)
        primary_clustering_data["num_points"] = len(points)

        if primary_clustering_data["num_points"] >= 2:
            distances = pdist(points)
            primary_clustering_data["mean_distance"] = np.mean(distances)
            primary_clustering_data["std_distance"] = np.std(distances)

        else:
            primary_clustering_data["mean_distance"] = 0
            primary_clustering_data["std_distance"] = 0
        
        primary_clustering_features = np.array([primary_clustering_data["num_points"],
                                            primary_clustering_data["mean_distance"],
                                            primary_clustering_data["std_distance"]])
        
        primary_clustering_model = joblib.load('models/primary_cluster_pipeline.pkl')
        param_states[3]["value"] = int(primary_clustering_model.predict([primary_clustering_features])[0])
        print(f"Predicted Primary Clustering Threshold: {param_states[3]["value"]}")
    clustered_pts = cluster_points(intersections, threshold=param_states[3]["value"]) #30 -70

    cv2.imshow("Primary Intersections", img3)

    pts = np.array(clustered_pts)
    src_points = find_board_corner(pts)
    cv2.circle(img3, tuple(map(int, src_points[0])), 5, (0, 0, 255), -1)    # top-left
    cv2.circle(img3, tuple(map(int, src_points[1])), 5, (255, 255, 0), -1)    # top-right
    cv2.circle(img3, tuple(map(int, src_points[2])), 5, (255, 255, 0), -1)    # bottom-right
    cv2.circle(img3, tuple(map(int, src_points[3])), 5, (255, 255, 0), -1)  # bottom-left


    
    cv2.imshow("Board Corners", img3)
    return src_points,primary_hough_training_data, primary_clustering_data

def secondary_processing(warped):
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blurred_warped = cv2.GaussianBlur(warped_gray, (5, 5), 0)
    edges_warped = cv2.Canny(blurred_warped, 50, 150, apertureSize=3)
    #cv2.imshow("Warped Edges", edges_warped)

    if (param_states[0]["value"] == 0):
        secondary_hough_training_data["mean_intensity"] = np.mean(edges_warped)
        secondary_hough_training_data["std_intensity"] = np.std(edges_warped)
        secondary_hough_training_data["edge_density"] = np.sum(edges_warped > 0) / (edges_warped.shape[0] * edges_warped.shape[1])

        #Gradient
        sobel_x_warped = cv2.Sobel(warped_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y_warped = cv2.Sobel(warped_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude_warped = np.sqrt(sobel_x_warped**2 + sobel_y_warped**2)
        secondary_hough_training_data["gradient_mean"] = np.mean(gradient_magnitude_warped)
        secondary_hough_training_data["gradient_std"] = np.std(gradient_magnitude_warped)

        #Entropy
        secondary_hough_training_data["entropy"] = shannon_entropy(warped_gray)

        #Aspect ratio
        secondary_hough_training_data["aspect_ratio"] = warped_gray.shape[1] / warped_gray.shape[0]

        secondary_hough_features = np.array([secondary_hough_training_data["mean_intensity"],
                                            secondary_hough_training_data["std_intensity"],
                                            secondary_hough_training_data["edge_density"],
                                            secondary_hough_training_data["gradient_mean"],
                                            secondary_hough_training_data["gradient_std"],
                                            secondary_hough_training_data["entropy"],
                                            secondary_hough_training_data["aspect_ratio"]])
        secondary_hough_model = joblib.load('models/secondary_hough_pipeline.pkl')
        param_states[0]["value"] = int(secondary_hough_model.predict([secondary_hough_features])[0])
        print(f"Predicted Secondary Hough Threshold: {param_states[0]['value']}")

    lines_warped = cv2.HoughLines(edges_warped, 1, np.pi / 180, param_states[0]["value"]) # 150 - 220
    img4 = warped.copy()
    if lines_warped is None:
        print("No lines found in warped image")
        exit()
    
    for rho_theta in lines_warped:
        rho, theta = rho_theta[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img4, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("Secondary Detected Lines 1", img4)

    wrapped_vertical_lines = []
    wrapped_horizontal_lines = []

    for rho_theta in lines_warped:
        rho, theta = rho_theta[0]
        angle = np.degrees(theta)
        if abs(angle) < 30 or abs(angle - 180) < 30:
            wrapped_vertical_lines.append((rho, theta))
        elif abs(angle - 90) < 30:
            wrapped_horizontal_lines.append((rho, theta))
    
    wrapped_intersections = []
    for vrho, vtheta in wrapped_vertical_lines:
        for hrho, htheta in wrapped_horizontal_lines:
            pt = compute_intersection(vrho, vtheta, hrho, htheta)
            if pt is not None:
                x, y = pt
                if 0 <= x < warped.shape[1] and 0 <= y < warped.shape[0]:
                    wrapped_intersections.append((x, y))

    if (param_states[1]["value"] == 0):
        secondary_clustering_data["num_points"] = len(wrapped_intersections)
        if secondary_clustering_data["num_points"] >= 2:
            distances = pdist(np.array(wrapped_intersections))
            secondary_clustering_data["mean_distance"] = np.mean(distances)
            secondary_clustering_data["std_distance"] = np.std(distances)
        
        secondary_clustering_features = np.array([secondary_clustering_data["num_points"],
                                                secondary_clustering_data["mean_distance"],
                                                secondary_clustering_data["std_distance"]])
        secondary_clustering_model = joblib.load('models/secondary_cluster_pipeline.pkl')
        param_states[1]["value"] = int(secondary_clustering_model.predict([secondary_clustering_features])[0])
        print(f"Predicted Secondary Clustering Threshold: {param_states[1]['value']}")

    cluster_wrapped_pts = cluster_points(wrapped_intersections, threshold=param_states[1]["value"]) #40 - 90

     # draw clusters
    for pt in cluster_wrapped_pts:
        cv2.circle(img4, pt, 4, (255, 0, 0), -1)
    cv2.imshow("Warped Intersections", img4)
    return cluster_wrapped_pts,img4

if __name__ == "__main__":
    image = cv2.imread('dataset/testing-images/10.jpg') # board image path
    exit = 0
    exit1 = 0
    exit2 = 0

    while exit == 0:

        while exit1 == 0:
            src_points, primary_hough_training_data, primary_clustering_data = preprocess_image(image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            decision = input("Is the board orientation correct? (y/n): ").strip().lower()
            if decision == 'y':
                cv2.destroyAllWindows()
                break
            cv2.destroyAllWindows()
            param_states[2]["value"] = int(input(f"Enter Hough threshold (current: {param_states[2]['value']}): ") or param_states[2]["value"])
            param_states[3]["value"] = int(input(f"Enter clustering threshold (current: {param_states[3]['value']}): ") or param_states[3]["value"])


        board_size = 800
        dst_points = np.float32([[10, 10], [board_size - 10, 10], [board_size - 10, board_size - 10], [10, board_size - 10]])
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(image, M, (board_size, board_size)) 
        cv2.imshow("Warped Board", warped)


        # Second processing
        cluster_wrapped_pts, img4 = secondary_processing(warped)
        sorted_grid = sort_into_grid(cluster_wrapped_pts, rows=9, cols=9)
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
            cluster_wrapped_pts, img4 = secondary_processing(warped)
            sorted_grid = sort_into_grid(cluster_wrapped_pts, rows=9, cols=9)

        print("Final Sorted Grid Size:", len(sorted_grid), "x", len(sorted_grid[0]) if sorted_grid else 0)
        vis_squares = warped.copy()

        for i in range(8):
            for j in range(8):
                # 4 corners of the square
                tl = sorted_grid[i][j] 
                tr = sorted_grid[i][j+1]
                br = sorted_grid[i+1][j+1]
                bl = sorted_grid[i+1][j]

                # Draw square outline
                pts = np.array([tl, tr, br, bl], np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_squares, [pts], isClosed=True, color=(0, 255, 255), thickness=1)

                # Add label at center
                cx = int((tl[0] + br[0]) / 2)
                cy = int((tl[1] + br[1]) / 2)
                label = f"{chr(ord('A') + i)}{8-j}"
                cv2.putText(vis_squares, label, (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.4, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow("Squares Visualization", vis_squares)
        cv2.waitKey(0)
        decision = input("Does the grid look correct? (y/n): " ).strip().lower()
        if decision == 'y':
            exit1 = 1
        cv2.destroyAllWindows()
        
        lastDecision = input("Is this final processing correct? (y/n): ").strip().lower()
        if lastDecision == 'y':
            exit = 1 
    
    #result = ""
    fen = ""
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
            confidence = np.max(predict)
            #result += f"{chr(ord('A') + i)}{8-j}: {predict_class}, {confidence}\n"
            if predict_class == "1":
                ones += 1
            else:
                fen += f"{ones}" if ones > 0 else ""
                ones = 0
                fen += predict_class

        if ones > 0:
            fen += f"{ones}"
            request_html += f"{ones}"
            ones = 0
        fen += "/"
    fen = fen[:-1] + " w KQkq - 0 1"
   
    
    print(f"Request URL: {request_html}")
    #result += f"FEN: {fen}\n"
    #print(f"{result}")
    print(f"FEN: {fen}")
    cv2.destroyAllWindows()
    
## © 2025 All rights reserved. Jared Aung
