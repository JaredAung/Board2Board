import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist
from collections import defaultdict
from skimage.measure import shannon_entropy
import joblib

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

def preprocess_image(image, primary_houghline, primary_clustering):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1) # could be 1 
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Mean and std intensity
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
    if primary_houghline == 0:
        primary_hough_features = np.array([primary_hough_training_data["mean_intensity"],
                                        primary_hough_training_data["std_intensity"],
                                        primary_hough_training_data["edge_density"],
                                        primary_hough_training_data["gradient_mean"],
                                        primary_hough_training_data["gradient_std"],
                                        primary_hough_training_data["entropy"],
                                        primary_hough_training_data["aspect_ratio"]])

        primary_hough_model = joblib.load('models/primary_hough_pipeline.pkl')
        primary_houghline = int(primary_hough_model.predict([primary_hough_features])[0])
        print(f"Predicted Primary Hough Threshold: {primary_houghline}")


    lines = cv2.HoughLines(edges, 1, np.pi / 180, primary_houghline) # 100 - 150 
    if lines is None:
        print("No lines found")
        exit()
    primary_hough_training_data["hough_threshold"] = primary_houghline
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
        if abs(angle) < 30 or abs(angle - 180) < 30: 
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
                    
    primary_clustering_data["cluster_threshold"] = primary_clustering
    points = np.array(intersections)
    primary_clustering_data["num_points"] = len(points)

    if primary_clustering_data["num_points"] >= 2:
        distances = pdist(points)
        primary_clustering_data["mean_distance"] = np.mean(distances)
        primary_clustering_data["std_distance"] = np.std(distances)

    else:
        primary_clustering_data["mean_distance"] = 0
        primary_clustering_data["std_distance"] = 0

    if primary_clustering == 0:
        primary_clustering_features = np.array([primary_clustering_data["num_points"],
                                            primary_clustering_data["mean_distance"],
                                            primary_clustering_data["std_distance"]])
        
        primary_clustering_model = joblib.load('models/primary_cluster_pipeline.pkl')
        primary_clustering = int(primary_clustering_model.predict([primary_clustering_features])[0])
        print(f"Predicted Primary Clustering Threshold: {primary_clustering}")
        primary_clustering_data["cluster_threshold"] = primary_clustering
    clustered_pts = cluster_points(intersections, threshold=primary_clustering) #30 -70

    cv2.imshow("Primary Intersections", img3)

    pts = np.array(clustered_pts)
    src_points = find_board_corner(pts)
    cv2.circle(img3, tuple(map(int, src_points[0])), 5, (0, 0, 255), -1)    # top-left
    cv2.circle(img3, tuple(map(int, src_points[1])), 5, (255, 255, 0), -1)    # top-right
    cv2.circle(img3, tuple(map(int, src_points[2])), 5, (255, 255, 0), -1)    # bottom-right
    cv2.circle(img3, tuple(map(int, src_points[3])), 5, (255, 255, 0), -1)  # bottom-left


    
    cv2.imshow("Board Corners", img3)
    
    return src_points, primary_hough_training_data, primary_clustering_data

def warp_board_by_corners(src_points, image):
    board_size = 800
    dst_points = np.float32([[10, 10], [board_size - 10, 10], [board_size - 10, board_size - 10], [10, board_size - 10]])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, M, (board_size, board_size)) 
    cv2.imshow("Warped Board", warped)
    return warped
    
