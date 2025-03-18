import yaml
import json
import os
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import cv2

def load_camera_intrinsics():
    with open('config/camera_intrinsics.json', 'r') as f:
        return json.load(f)
    
def load_category_colors():
    with open('config/category_colors.json', 'r') as f:
        data = json.load(f)
    
    # Reconstruct the original dictionary format
    return {int(k): (v['color'], v['name']) for k, v in data.items()}

def load_config(config_path):
    """Load the YAML configuration."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration file: {config_path}. Error: {str(e)}")
    return config

def colorize_point_cloud_photorealisitc(lidar_points, image, transformed_points):
    """Colorize the point cloud based on the image colors with correct RGB values."""
    print("Colorizing the point cloud with RGB values.")

    Xc, Yc, Zc = transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2]
    camera_intrinsics = load_camera_intrinsics()
    
    # Step 1: Find valid points that are in front of the camera (Z > 0)
    valid_indices = np.where(Zc > 0)[0]  # Get indices directly to maintain order
    Xc, Yc, Zc = Xc[valid_indices], Yc[valid_indices], Zc[valid_indices]

    # Step 2: Project the valid points into the image plane
    x_proj = (camera_intrinsics["fx"] * Xc / Zc + camera_intrinsics["cx"]).astype(int)
    y_proj = (camera_intrinsics["fy"] * Yc / Zc + camera_intrinsics["cy"]).astype(int)


    # Step 3: Find points that fall within the image bounds
    inside_image = (x_proj >= 0) & (x_proj < image.shape[1]) & (y_proj >= 0) & (y_proj < image.shape[0])
    
    # Step 4: Apply the mask to retain only valid points (ensuring order is kept)
    valid_indices = valid_indices[inside_image]  # Filter indices based on image bounds
    x_proj, y_proj = x_proj[inside_image], y_proj[inside_image]

    # Step 5: Extract RGB colors
    colors = image[y_proj, x_proj]

    colored_points = [
        (lidar_points[i][0], lidar_points[i][1], lidar_points[i][2], 
        int(colors[j][2]), int(colors[j][1]), int(colors[j][0]))  # Store as BGR
        for j, i in enumerate(valid_indices)
    ]

    print("Successfully colorized {} points.".format(len(colored_points)))
    return colored_points, valid_indices

def colorize_point_cloud_semantic(lidar_points, image, transformed_points, semantic_predictions):
    """Colorize the point cloud based on the semantic segmentation mask and store class labels."""
    print("Colorizing the point cloud using semantic segmentation.")

    Xc, Yc, Zc = transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2]

    # Step 1: Keep only forward-facing points
    valid_indices = np.where(Zc > 0)[0]  
    Xc, Yc, Zc = Xc[valid_indices], Yc[valid_indices], Zc[valid_indices]
    camera_intrinsics = load_camera_intrinsics()
    # Step 2: Project points into 2D image (original resolution)
    x_proj = (camera_intrinsics["fx"] * Xc / Zc + camera_intrinsics["cx"]).astype(int)
    y_proj = (camera_intrinsics["fy"] * Yc / Zc + camera_intrinsics["cy"]).astype(int)

    # Step 3: Find points inside the original image bounds
    img_h, img_w = image.shape[:2]  # Get original image size
    inside_image = (x_proj >= 0) & (x_proj < img_w) & (y_proj >= 0) & (y_proj < img_h)

    valid_indices = valid_indices[inside_image]
    x_proj, y_proj = x_proj[inside_image], y_proj[inside_image]

    # **Use semantic predictions to get class labels**
    semantic_labels = semantic_predictions[y_proj, x_proj]  # Get semantic class at projection

    # Step 4: Assign RGB color from category colors & store class labels
    colored_points = []
    class_labels = []
    CATEGORY_COLORS = load_category_colors()

    for j, i in enumerate(valid_indices):
        class_id = int(semantic_labels[j])  # Get class label
        color = CATEGORY_COLORS[class_id][0]  # Get color from CATEGORY_COLORS

        # Store point + color
        colored_points.append((
            lidar_points[i][0], lidar_points[i][1], lidar_points[i][2], 
            color[0], color[1], color[2]
        ))

        # Store class label separately
        class_labels.append(class_id)

    print(f"Successfully colorized {len(colored_points)} points using semantic segmentation.")
    return colored_points, class_labels, valid_indices

def overlay_points_on_black_image(u, v, image):
    """
    Draw projected points onto a black image, using the corresponding color from the original image.
    Publishes the image as a ROS topic.
    """
    img_h, img_w, _ = image.shape
    black_image = np.zeros((img_h, img_w, 3), dtype=np.uint8)  # Create black image

    for i in range(len(u)):
        u_i = int(u[i])
        v_i = int(v[i])

        if 0 <= u_i < img_w and 0 <= v_i < img_h:
            # Get the actual color from the original image at this point
            color = image[v_i, u_i].tolist()  # Extract RGB color from the image
            cv2.circle(black_image, (u_i, v_i), 2, color, -1)  # Draw point with same color

    # Convert black image to ROS format and publish
    print("Published projected LiDAR points onto a black image.")
    return black_image

def save_pointcloud_to_txt(frame_count, transformed_points, colored_points, valid_indices, timestamp, lidar_points, output_txt_dir, mode):
    """Saves the processed colorized LiDAR-to-Camera projected point cloud to a text file."""
    folder_path = os.path.join(output_txt_dir, mode)
    os.makedirs(folder_path, exist_ok=True)  # Ensure folder exists
    
    # ✅ Corrected filename generation
    filename = os.path.join(folder_path, f"{mode}_pcd_{frame_count:06d}.txt")

    # Extract valid LiDAR points, intensity, ring, and time fields
    # filtered_lidar_points_ = transformed_points[valid_indices]  # Use transformed LiDAR-to-Camera points
    filtered_lidar_points = lidar_points[:, :3][valid_indices]  # Use original LiDAR-to-Camera points
    # import pdb;pdb.set_trace()
    filtered_intensity = lidar_points[valid_indices, 3]  # Extract intensity
    filtered_ring = lidar_points[valid_indices, 4]  # Extract ring index
    filtered_time = lidar_points[valid_indices, 5]  # Extract time within rotation

    with open(filename, "w") as f:
        for i in range(len(filtered_lidar_points)):
            x, y, z = filtered_lidar_points[i]  # Use transformed (valid) LiDAR points
            intensity = filtered_intensity[i]  # Get intensity
            ring = int(filtered_ring[i])  # Get ring index (integer)
            time_within_rotation = filtered_time[i]  # Get time within rotation
            
            # Extract RGB values
            _, _, _, r, g, b = colored_points[i]  

            # Save all data in the requested format
            f.write(f"{timestamp:.6f},{x:.6f},{y:.6f},{z:.6f},{intensity:.2f},{ring},{time_within_rotation:.6f},{r},{g},{b}\n")




def remove_water_using_ransac(pcd, distance_threshold=0.3, ransac_n=3, num_iterations=1000):
    """
    Removes flat surfaces like water using RANSAC plane segmentation.
    :param pcd: Open3D point cloud object
    :param distance_threshold: Max distance a point can be from the plane to be considered an inlier
    :param ransac_n: Number of points used to estimate the plane
    :param num_iterations: Number of iterations to run RANSAC
    :return: Filtered point cloud without the detected plane (water)
    """

    print("Applying RANSAC to remove water reflections...")

    # Segment the dominant plane (likely the water surface)
    plane_model, inlier_indices = pcd.segment_plane(distance_threshold=distance_threshold,
                                                    ransac_n=ransac_n,
                                                    num_iterations=num_iterations)
    
    # Extract inliers (plane points) and outliers (remaining points)
    inlier_cloud = pcd.select_by_index(inlier_indices)  # This contains the plane (water) points
    outlier_cloud = pcd.select_by_index(inlier_indices, invert=True)  # These are the remaining points
    
    print(f"Removed {len(inlier_indices)} points belonging to the water surface.")
    
    return outlier_cloud

def generate_boat_trajectory_pcd(odometry_data):
    """Generate a point cloud representing the boat's trajectory from odometry data."""
    
    # Extract only position data from odometry
    trajectory_points = np.array([data[1] for data in odometry_data])  # XYZ positions of the boat

    # Assign a fixed color (e.g., BLUE [0, 0, 1]) for trajectory points
    trajectory_colors = np.full((trajectory_points.shape[0], 3), [0, 0, 1])  # Blue color for trajectory

    # Convert to Open3D format
    trajectory_pcd = o3d.geometry.PointCloud()
    trajectory_pcd.points = o3d.utility.Vector3dVector(trajectory_points)
    trajectory_pcd.colors = o3d.utility.Vector3dVector(trajectory_colors.astype(np.float32))

    return trajectory_pcd  # Return the trajectory point cloud

def apply_color_map( predictions):
    """Maps prediction labels to their respective CATEGORY_COLORS"""
    height, width = predictions.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    CATEGORY_COLORS = load_category_colors()

    for label, (color, name) in CATEGORY_COLORS.items():
        color_mask[predictions == label] = color  # Assign RGB color

    return color_mask

def compute_map_entropy(pcd, num_bins=50):
    """
    Compute entropy of a point cloud based on point density.
    :param pcd: Open3D point cloud object
    :param num_bins: Number of bins for histogram
    :return: Entropy value and nearest distance histogram
    """
    points = np.asarray(pcd.points)

    # Compute nearest neighbor distances
    tree = KDTree(points)
    nearest_distances, _ = tree.query(points, k=2)  # Find nearest neighbor for each point
    nearest_distances = nearest_distances[:, 1]  # Ignore self-distance

    # Compute histogram of nearest distances
    hist, bins = np.histogram(nearest_distances, bins=num_bins, density=True)

    # Compute entropy from histogram
    prob = hist / hist.sum()
    entropy = -np.sum(prob * np.log2(prob + 1e-9))  # Avoid log(0)

    print(f"Map Entropy: {entropy:.4f}")
    return entropy, nearest_distances

def plot_entropy_distribution(nearest_distances, num_bins=50):
    """
    Plot histogram of nearest neighbor distances to understand entropy.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(nearest_distances, bins=num_bins, color='blue', alpha=0.7, density=True)
    plt.xlabel("Nearest Neighbor Distance")
    plt.ylabel("Frequency")
    plt.title("Distribution of Point Distances (Map Entropy Analysis)")
    plt.grid(True)
    plt.show()

def filter_points_by_class(pcd, class_label):
    indices = np.where(pcd.semantic_labels == class_label)[0]
    filtered_pcd = pcd.select_by_index(indices)
    return filtered_pcd




def save_filtered_odometry(filtered_odometry, odometry_txt_path):
    """
    Saves the filtered trajectory data corresponding to processed LiDAR frames.
    
    Ensures the trajectory file contains only odometry timestamps matching 
    the .txt files in the "semantic" or "photo" folders.
    """
    

    with open(odometry_txt_path, "w") as f:
        for timestamp, position, orientation in filtered_odometry:
            # ✅ Save in format: timestamp, x, y, z, qx, qy, qz, qw
            f.write(f"{timestamp:.6f},{position[0]:.6f},{position[1]:.6f},{position[2]:.6f},"
                    f"{orientation[0]:.6f},{orientation[1]:.6f},{orientation[2]:.6f},{orientation[3]:.6f}\n")

    print(f"✅ Filtered trajectory (actually used values) saved to {odometry_txt_path}")
