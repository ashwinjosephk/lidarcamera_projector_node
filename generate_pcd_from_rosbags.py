import rosbag
import numpy as np
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
import tf.transformations as tf
from cv_bridge import CvBridge
import cv2
import os
import shutil
import time
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch
from PIL import Image


# Camera Intrinsics (to be dynamically set instead of hardcoded values)
camera_intrinsics = {
    "fx": 1089.8,
    "fy": 1086,
    "cx": 1150.3,
    "cy": 638.8
}



def colorize_point_cloud_photorealisitc(lidar_points, image, transformed_points):
    """Colorize the point cloud based on the image colors with correct RGB values."""
    print("Colorizing the point cloud with RGB values.")

    Xc, Yc, Zc = transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2]
    
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

def save_pointcloud_to_txt(frame_count, transformed_points, colored_points, valid_indices, timestamp, lidar_points):
    """Saves the processed colorized LiDAR-to-Camera projected point cloud to a text file."""
    
    filename = os.path.join(output_txt_dir, f"{output_txt_dir}_pcd_{frame_count:06d}.txt")
    
    # Extract valid LiDAR points, intensity, ring, and time fields
    filtered_lidar_points = transformed_points[valid_indices]  # Use transformed LiDAR-to-Camera points
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
            f.write(f"{x:.6f},{y:.6f},{z:.6f},{intensity:.2f},{ring},{time_within_rotation:.6f},{r},{g},{b}\n")

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

def voxel_grid_downsample(pcd, voxel_size=0.2):
    """
    Apply voxel grid downsampling to reduce the number of points.
    :param pcd: Open3D point cloud object
    :param voxel_size: The size of each voxel (lower values retain more detail)
    :return: Downsampled point cloud
    """
    print(f"Applying Voxel Grid Downsampling with voxel size: {voxel_size}")
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"Reduced point count from {len(pcd.points)} to {len(downsampled_pcd.points)}")
    return downsampled_pcd


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

def apply_color_map( predictions):
    """Maps prediction labels to their respective CATEGORY_COLORS"""
    height, width = predictions.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for label, (color, name) in CATEGORY_COLORS.items():
        color_mask[predictions == label] = color  # Assign RGB color

    return color_mask

def semantic_segmentation_inference(image, model, processor):
    """Run semantic segmentation inference using Segformer model."""
    original_height, original_width = image.shape[:2]
    
    # Convert image to PIL for processing
    image_pil = Image.fromarray(image)

    # Process image with Segformer processor
    pixel_values = processor(images=image_pil, return_tensors="pt").pixel_values.to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    print("Semantic Segmentation inference done!")

    # Get predictions
    logits = outputs.logits
    predictions = logits.argmax(dim=1).squeeze().cpu().numpy()  # Get highest probability class

    # **Resize segmentation predictions back to original image size**
    predictions_resized = cv2.resize(predictions, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

    # Convert predictions to colored mask
    color_mask = apply_color_map(predictions_resized)
    print("Color Map applied")

    return predictions_resized, color_mask

def filter_pcd_by_class(pcd, class_labels_file, selected_classes, CATEGORY_COLORS):
    """
    Filter points in the point cloud based on selected semantic classes.
    :param pcd: Open3D Point Cloud
    :param class_labels_file: Path to saved class labels `.npy` file
    :param selected_classes: List of class indices to keep
    :return: Filtered point cloud
    """
    class_labels = np.load(class_labels_file)  # Load class labels
    class_labels = class_labels.flatten()  # Ensure it's 1D

    # Get indices of points belonging to selected classes
    indices = np.where(np.isin(class_labels, selected_classes))[0]

    # Create filtered point cloud
    filtered_pcd = pcd.select_by_index(indices)

    selected_class_names = [CATEGORY_COLORS[i][1] for i in selected_classes if i in CATEGORY_COLORS]

    print(f"✅ Filtered PCD contains {len(indices)} points from selected classes {selected_class_names}.")
    
    return filtered_pcd

def compute_depth_from_stereo(left_image, right_image, camera_intrinsics):
    """
    Compute a depth image from stereo images using OpenCV's StereoSGBM.
    :param left_image: Left rectified image (BGR or grayscale)
    :param right_image: Right rectified image (BGR or grayscale)
    :param camera_intrinsics: Dictionary with 'fx' (focal length in pixels)
    :return: Depth image in meters (same size as input images)
    """
    baseline = 0.12  # Baseline of ZED2i stereo camera in meters

    # Convert images to grayscale if needed
    if len(left_image.shape) == 3:
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_image
        right_gray = right_image

    # Stereo matcher settings
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,  # Must be a multiple of 16
        blockSize=9,
        P1=8 * 3 * 9 ** 2,
        P2=32 * 3 * 9 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    # Compute disparity
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0  # Normalize

    # Avoid division by zero (replace invalid disparity values with small nonzero value)
    disparity[disparity <= 0] = 0.1

    # Compute depth using Depth = (fx * B) / disparity
    depth = (camera_intrinsics["fx"] * baseline) / disparity

    return depth



if __name__=="__main__":
    # Paths to ROS bags and output file

    '''
    Filenames:
    7_anlegen_80m_100kmph_BTUwDLR
    9_anlegen_80m_100kmph_BTUwDLR
    17_straight_200m_100kmph_BTUwDLR
    29_bridgecurve_80m_100kmph_BTUwDLR
    30_bridgecurve_80m_100kmph_BTUwDLR
    37_curvepromenade_160m_100kmph_BTUwDLR
    53_schleuseeinfahrt_20m_100kmph_BTUwDLR

    '''


    lidar_bag_base_name = "53_schleuseeinfahrt_20m_100kmph_BTUwDLR"

    #Paths
    lidar_bag_path = "/home/knadmin/Ashwin/AURORA_dataset/Segmented3DMap/DATA/{}/{}.bag".format(lidar_bag_base_name,lidar_bag_base_name)
    odometry_bag_path = "/home/knadmin/Ashwin/AURORA_dataset/Segmented3DMap/DATA/{}/{}_trajectory.bag".format(lidar_bag_base_name,lidar_bag_base_name)
    original_bag_images_dump_folder = "/home/knadmin/Ashwin/AURORA_dataset/Segmented3DMap/DATA/{}/{}_images/images".format(lidar_bag_base_name,lidar_bag_base_name)
    output_txt_dir = "/home/knadmin/Ashwin/AURORA_dataset/Segmented3DMap/DATA/{}".format(lidar_bag_base_name)

    output_folder = os.path.join(output_txt_dir,"output_pcd")
    os.makedirs(output_folder, exist_ok=True)
    output_pcd_semantic_file = "{}/{}_trajectory_output_map_semantic.ply".format(output_folder,lidar_bag_base_name)
    output_pcd_photo_file = "{}/{}_trajectory_output_map_photo.ply".format(output_folder,lidar_bag_base_name)
    output_trajectory_pcd_file = "{}/{}_trajectory_output_map_trajectory.ply".format(output_folder,lidar_bag_base_name)
    filtered_pcd_path = "{}/{}_filtered_pcd.ply".format(output_folder,lidar_bag_base_name)

    #TOpic names
    lidar_topic = "/VLP32/velodyne_points"
    left_image_topic = "/zed2i/zed_node/left/image_rect_color/compressed"
    right_image_topic = "/zed2i/zed_node/right/image_rect_color/compressed"
    odometry_topic = "/lio_sam/mapping/odometry"
    

    trained_models_Save_path = "/home/knadmin/Ashwin/Semantic_labelled_by_Lukas_Hosch/trained_models"
    model_name = "segformer-best_6classes_aug_adjustable_lr_customweights"
    model_save_path = os.path.join(trained_models_Save_path,model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
    # Load the best saved model before inference
    model = SegformerForSemanticSegmentation.from_pretrained(model_save_path).to(device)
    model.eval()  # Ensure model is in evaluation mode
    model.to(device)
    CATEGORY_COLORS = {
    0: ([135, 206, 250], "Sky"),        # Sky
    1: ([0, 191, 255], "Water"),          # Water
    2: ([50, 205, 50], "Vegetation"),     # Vegetation
    3: ([34, 139, 34], "Riverbank"),       # Riverbank
    4: ([184, 134, 11], "Bridge"),        # Bridge
    5: ([157, 0, 255], "Other")         # Other
    }
    selected_classes = [1, 2, 4]  #for filtering specific classes in the filtered pointcloud

    trajectory_flag = True
    apply_ransac_flag = False
    compute_map_entropy_flag = False
    downsample_pcd_flag = False
    save_images_flag = False

    frame_count = 0  # Counter for frame-based naming

    # if os.path.exists(output_txt_dir):
    #     shutil.rmtree(output_txt_dir)  
    os.makedirs(output_txt_dir, exist_ok=True)  # Create new empty folder
    
    # cv2.namedWindow("Original left Camera Image",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Original right Camera Image",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Original Image with lidar points in red projected", cv2.WINDOW_NORMAL) 
    cv2.namedWindow("Only Lidar points with Photo colour", cv2.WINDOW_NORMAL) 
    cv2.namedWindow("Semantic Image", cv2.WINDOW_NORMAL) 
    cv2.namedWindow("Alpha Blended Image",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth Image",cv2.WINDOW_NORMAL)

    # LiDAR to Camera Transformation Matrix
    R = np.array([[-0.0086, 0.0068, 0.9999],
                [-1.0000, -0.0006, -0.0086],
                [0.0006, -1.0000, 0.0068]])
    translation = np.array([0.0441, 0.0649, -0.0807])  # My estimates

    def rigid_transform(points, R, translation):
        """Apply rigid transformation to the point cloud."""
        return np.dot(points, R) + translation

    bridge = CvBridge()

    start_time = time.time()

    # Load odometry from ROS bag with high-precision timestamps
    odometry_data = []
    odometry_bag = rosbag.Bag(odometry_bag_path, "r")
    for topic, msg, t in odometry_bag.read_messages(topics=[odometry_topic]):
        timestamp = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9  # High-precision timestamp
        position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        orientation = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        odometry_data.append((timestamp, position, orientation))
    odometry_bag.close()

    # Sort odometry data by timestamp
    odometry_data.sort(key=lambda x: x[0])
    odometry_timestamps = [x[0] for x in odometry_data]

    # Process LiDAR and Image data together
    pcd_points = []
    photo_colors_list = []
    semantic_colors_list = []
    intensity_list = []
    ring_list = []
    class_labels_list = []

    lidar_bag = rosbag.Bag(lidar_bag_path, "r")
    left_image = None  # Initialize image storage
    right_image = None  # Initialize image storage
    depth_image = None

    points_counter =0
    image_counter = 0

    for topic, msg, t in lidar_bag.read_messages(topics=[lidar_topic, left_image_topic, right_image_topic]):
        timestamp = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9  # High-precision timestamp
        topic_info = lidar_bag.get_type_and_topic_info()
        total_lidar_messages = topic_info.topics[lidar_topic].message_count

        
        if topic == left_image_topic:
            left_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            image_counter+=1
            # cv2.imshow("Original left Camera Image", left_image)
            if save_images_flag:
                os.makedirs(original_bag_images_dump_folder, exist_ok=True)
                img_path = os.path.join(original_bag_images_dump_folder,"image_{}.png".format(timestamp))
                cv2.imwrite(img_path,left_image)
            continue  # Store the latest image and move on

        if topic == right_image_topic:
            right_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            print("right_image.shape",right_image.shape)
            cv2.imshow("Original right Camera Image", right_image)
            # Ensure we have both left and right images before computing depth
            if left_image is not None:
                depth_image = compute_depth_from_stereo(left_image, right_image, camera_intrinsics)
                print("depth image computed")

                # Normalize depth for visualization (convert meters to grayscale)
                depth_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)  # Use JET colormap for better visualization
                print("depth image colored")

                # Show the depth image
                cv2.imshow("Depth Image", depth_colored)
                # cv2.waitKey(1)
        
        if topic == lidar_topic and left_image is not None:
            # Find the closest odometry timestamp
            closest_idx = np.argmin(np.abs(np.array(odometry_timestamps) - timestamp))
            closest_time, position, orientation = odometry_data[closest_idx]

            frame_inference_start_time = time.time()
            print("**************************")
            print("Frame counter: {}/{}".format(frame_count,total_lidar_messages))
            
            print(f"Processing LiDAR at {timestamp} -> Closest odometry at {closest_time} (Δt = {abs(closest_time - timestamp):.3f}s)")
            
            # Convert LiDAR message to numpy array- every single lidar point in this topic
            lidar_points = np.array(list(pc2.read_points(
                    msg, field_names=("x", "y", "z", "intensity", "ring", "time"), skip_nans=True)))
            
            # Extract x, y, z, intensity, and ring separately
            points = lidar_points[:, :3]
            intensity = lidar_points[:, 3]
            ring = lidar_points[:, 4]


            
            # Apply LiDAR to Camera transformation for lidar to camera projection- every point still there
            transformed_points = rigid_transform(lidar_points[:, :3], R, translation)

            Xc, Yc, Zc = transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2]

            print("Number of valid points before filtering: {} (In Total Points)".format(len(Xc)))

            valid_indices = Zc > 0
            Xc, Yc, Zc = Xc[valid_indices], Yc[valid_indices], Zc[valid_indices]
            print("Number of valid points after filtering: {} (By taking points only with forward z axis that is direction camera faces)".format(len(Xc)))


            x_proj = (camera_intrinsics["fx"] * Xc / Zc + camera_intrinsics["cx"]).astype(int)
            y_proj = (camera_intrinsics["fy"] * Yc / Zc + camera_intrinsics["cy"]).astype(int)


            #checking if projected 2d points are within image boundary
            valid_points = (x_proj >= 0) & (x_proj < left_image.shape[1]) & (y_proj >= 0) & (y_proj < left_image.shape[0])
            count_trues_projected = np.sum(valid_points)
            print("Number of valid points after using points within image {} (and after z axes filtering)".format(count_trues_projected))

            x_proj, y_proj = x_proj[valid_points], y_proj[valid_points]

            # 1️⃣ Create the original image with red-projected points
            mask = np.zeros(left_image.shape[:2], dtype=np.uint8)
            image_with_red = left_image.copy()
            point_size = 2
            for x, y in zip(x_proj, y_proj):
                x_start, x_end = max(0, x - point_size), min(left_image.shape[1], x + point_size + 1)
                y_start, y_end = max(0, y - point_size), min(left_image.shape[0], y + point_size + 1)
                image_with_red[y_start:y_end, x_start:x_end] = [0, 0, 255]  # Red color
                mask[y_start:y_end, x_start:x_end] = 255  # White mask

            colored_points_photo, valid_indices  = colorize_point_cloud_photorealisitc(lidar_points, left_image, transformed_points)

            #code part to run inference of semantic segmentation
            semantic_predictions, semantic_color_mask = semantic_segmentation_inference(left_image, model, processor)
            semantic_color_mask = cv2.cvtColor(semantic_color_mask, cv2.COLOR_RGB2BGR)

            frame_inference_end_time = time.time()
            frame_inference_duration = frame_inference_end_time - frame_inference_start_time
            print("Per Frame inference Time",frame_inference_duration)

            #Part of the code to use for using semantic iamge color
            colored_points_semantic, class_labels, valid_indices = colorize_point_cloud_semantic(lidar_points, left_image, transformed_points, semantic_predictions)
            # class_labels_array = np.array(class_labels).reshape(-1, 1)
            

            # Overlay segmentation mask on original image
            alpha = 0.5
            overlayed_image = cv2.addWeighted(left_image, 1, semantic_color_mask, alpha, 0)


            # Save transformed + colorized LiDAR points to TXT
            # save_pointcloud_to_txt(frame_count, transformed_points, colored_points_rgb, valid_indices, timestamp, lidar_points)

            frame_count += 1  # Increment frame count
            
            valid_points = points[valid_indices] 
        

            # colors = np.zeros((points.shape[0], 3), dtype=np.uint8)

            print("checking length", len(colored_points_semantic), len(valid_indices))
            points_counter += len(valid_indices)
            # import pdb;pdb.set_trace()

            black_image = overlay_points_on_black_image(x_proj, y_proj, left_image)


            #part related to creating pointcloud
            # Get corresponding odometry transformation
            transformation = tf.quaternion_matrix(orientation)
            transformation[:3, 3] = position
        
            # Transform LiDAR points
            # ones = np.ones((points.shape[0], 1))
            ones = np.ones((valid_points.shape[0], 1))

            # homogenous_points = np.hstack((points, ones))
            homogenous_points = np.hstack((valid_points, ones))

            transformed_points = (transformation @ homogenous_points.T).T[:, :3]



            # Step 4: Extract valid colors (Expanded for debugging)
            semantic_colors = np.zeros((len(valid_indices), 3))  # Initialize color array
            photo_colors = np.zeros((len(valid_indices), 3))  # Initialize color array

            total_points = len(valid_indices)  # Total number of valid colorized points
            non_black_count = 0  # Counter for non-black points

            for idx, point in enumerate(colored_points_semantic):
                x, y, z, r, g, b = point  # Unpack each point with color values

                # Normalize RGB values (Open3D expects colors in range [0,1])
                r_norm = r / 255.0
                g_norm = g / 255.0
                b_norm = b / 255.0

                semantic_colors[idx] = [r_norm, g_norm, b_norm]
                # colors[idx] = [1, 0, 0]

            for idx, point in enumerate(colored_points_photo):
                x, y, z, r, g, b = point  # Unpack each point with color values

                # Normalize RGB values (Open3D expects colors in range [0,1])
                r_norm = r / 255.0
                g_norm = g / 255.0
                b_norm = b / 255.0

                photo_colors[idx] = [r_norm, g_norm, b_norm]
                # colors[idx] = [1, 0, 0]




        
            # Append to global lists
            pcd_points.append(transformed_points)
            semantic_colors = semantic_colors.astype(np.float32)
            photo_colors = photo_colors.astype(np.float32)
            semantic_colors_list.append(semantic_colors)
            photo_colors_list.append(photo_colors)
            intensity_list.append(intensity)
            ring_list.append(ring)
            class_labels_list.extend(class_labels) 



            cv2.imshow("Original Image with lidar points in red projected",image_with_red) 
            cv2.imshow("Only Lidar points with Photo colour", black_image) 
            cv2.imshow("Semantic Image", semantic_color_mask) 
            cv2.imshow("Alpha Blended Image", overlayed_image) 
            cv2.waitKey(50) 


    cv2.destroyAllWindows()
    print("************************")

    # Convert to Open3D format
    pcd_points = np.vstack(pcd_points)
    semantic_colors_list = np.vstack(semantic_colors_list)
    photo_colors_list = np.vstack(photo_colors_list)
    intensity_list = np.hstack(intensity_list)[:, None]  # Ensure correct shape
    ring_list = np.hstack(ring_list)[:, None]
    class_labels_array = np.array(class_labels_list).reshape(-1, 1)


    print(f"Total points saved in final PCD: {pcd_points.shape[0]}")
    print(f"Total points from valid indexes across entire rosbag: {points_counter}")
    print("image_counter",image_counter)


    # Create structured Open3D point cloud
    pcd_semantic = o3d.geometry.PointCloud()
    pcd_semantic.points = o3d.utility.Vector3dVector(pcd_points)
    pcd_semantic.colors = o3d.utility.Vector3dVector(semantic_colors_list)
    # pcd_semantic.class_labels = class_labels_array 
    


    pcd_photo = o3d.geometry.PointCloud()
    pcd_photo.points = o3d.utility.Vector3dVector(pcd_points)
    pcd_photo.colors = o3d.utility.Vector3dVector(photo_colors_list)
    

    # car_points = filter_points_by_class(pcd, car_class_label)

    if apply_ransac_flag:
        # Apply RANSAC to remove water reflections (ground-like surface)
        pcd_semantic = remove_water_using_ransac(pcd_semantic)
        pcd_photo = remove_water_using_ransac(pcd_photo)

    if compute_map_entropy_flag:
        # Compute entropy
        entropy, nearest_distances = compute_map_entropy(pcd_semantic)
        plot_entropy_distribution(nearest_distances)

    if downsample_pcd_flag:
        pcd_semantic = voxel_grid_downsample(pcd_semantic, voxel_size=0.2)  # Adjust voxel size for detail
        pcd_photo = voxel_grid_downsample(pcd_photo, voxel_size=0.2) 

    if trajectory_flag:
        # Generate trajectory point cloud for the boat's movement
        trajectory_pcd = generate_boat_trajectory_pcd(odometry_data)

        # Merge LiDAR and trajectory point clouds into one visualization
        combined_pcd_semantic = pcd_semantic + trajectory_pcd  # Combine both datasets
        combined_pcd_photo = pcd_photo + trajectory_pcd 


    end_time = time.time()
    time_taken = end_time - start_time
    print("Total TIme taken",time_taken)

    # Save the final point cloud
    print(f"Saving final Semantic 3D map to {output_pcd_semantic_file}")
    

    if trajectory_flag:

        o3d.io.write_point_cloud(output_pcd_semantic_file, combined_pcd_semantic)
        o3d.io.write_point_cloud(output_pcd_photo_file, combined_pcd_photo)
        o3d.io.write_point_cloud(output_trajectory_pcd_file, trajectory_pcd)
        np.save("class_labels.npy", class_labels_array)
        # Visualize both LiDAR map and Boat trajectory together




        o3d.visualization.draw_geometries([pcd_semantic], 
                                        window_name="Semantic 3D Map + Boat Trajectory",
                                        point_show_normal=False)
        o3d.visualization.draw_geometries([pcd_photo], 
                                        window_name="Photorealistic Map + Boat Trajectory",
                                        point_show_normal=False)
        o3d.visualization.draw_geometries([trajectory_pcd], 
                                        window_name="Boat Trajectory",
                                        point_show_normal=False)
        
        

        # ✅ Filter only selected classes and save separately
        filtered_pcd = filter_pcd_by_class(pcd_semantic, "class_labels.npy", selected_classes,CATEGORY_COLORS)
        o3d.visualization.draw_geometries([filtered_pcd], 
                                        window_name="Filtered Pointcloud",
                                        point_show_normal=False)
        o3d.io.write_point_cloud(filtered_pcd_path, filtered_pcd)
        print("✅ Filtered PCD saved as filtered_pcd.ply")

    else:
        o3d.io.write_point_cloud(output_pcd_semantic_file, pcd_semantic)
        o3d.io.write_point_cloud(output_pcd_photo_file, pcd_photo)
        # Visualize the final point cloud
        o3d.visualization.draw_geometries([pcd_semantic], window_name="Photorealistic 3D Map",
                                        point_show_normal=False)

    lidar_bag.close()








    #2432116



        




