import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist
from collections import defaultdict
from skimage.measure import shannon_entropy
import joblib

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

def secondary_processing(warped, secondary_houghline, secondary_clustering):
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blurred_warped = cv2.GaussianBlur(warped_gray, (5, 5), 0)
    edges_warped = cv2.Canny(blurred_warped, 50, 150, apertureSize=3)
    #cv2.imshow("Warped Edges", edges_warped)

    
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
    if secondary_houghline == 0:
        secondary_hough_features = np.array([secondary_hough_training_data["mean_intensity"],
                                            secondary_hough_training_data["std_intensity"],
                                            secondary_hough_training_data["edge_density"],
                                            secondary_hough_training_data["gradient_mean"],
                                            secondary_hough_training_data["gradient_std"],
                                            secondary_hough_training_data["entropy"],
                                            secondary_hough_training_data["aspect_ratio"]])
        secondary_hough_model = joblib.load('models/secondary_hough_pipeline.pkl')
        secondary_houghline = int(secondary_hough_model.predict([secondary_hough_features])[0])
        secondary_hough_training_data["hough_threshold"] = secondary_houghline
        print(f"Predicted Secondary Hough Threshold: {secondary_houghline}")

    lines_warped = cv2.HoughLines(edges_warped, 1, np.pi / 180, secondary_houghline) # 150 - 220
    img4 = warped.copy()
    if lines_warped is None:
        print("No lines found in warped image")
        exit()
    secondary_hough_training_data["hough_threshold"] = secondary_houghline

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

    
    secondary_clustering_data["num_points"] = len(wrapped_intersections)
    if secondary_clustering_data["num_points"] >= 2:
        distances = pdist(np.array(wrapped_intersections))
        secondary_clustering_data["mean_distance"] = np.mean(distances)
        secondary_clustering_data["std_distance"] = np.std(distances)

    if (secondary_clustering == 0):
        secondary_clustering_features = np.array([secondary_clustering_data["num_points"],
                                                secondary_clustering_data["mean_distance"],
                                                secondary_clustering_data["std_distance"]])
        secondary_clustering_model = joblib.load('models/secondary_cluster_pipeline.pkl')
        secondary_clustering = int(secondary_clustering_model.predict([secondary_clustering_features])[0])
        secondary_clustering_data["cluster_threshold"] = secondary_clustering
        print(f"Predicted Secondary Clustering Threshold: {secondary_clustering}")

    cluster_wrapped_pts = cluster_points(wrapped_intersections, threshold=secondary_clustering) #40 - 90
    secondary_clustering_data["cluster_threshold"] = secondary_clustering
     # draw clusters
    for pt in cluster_wrapped_pts:
        cv2.circle(img4, pt, 4, (255, 0, 0), -1)
    cv2.imshow("Warped Intersections", img4)
    return cluster_wrapped_pts,img4,secondary_hough_training_data,secondary_clustering_data


def visualize_grid(vis_squares,sorted_grid):
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

    return vis_squares

