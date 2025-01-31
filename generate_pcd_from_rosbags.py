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

# Camera Intrinsics (to be dynamically set instead of hardcoded values)
camera_intrinsics = {
    "fx": 1089.8,
    "fy": 1086,
    "cx": 1150.3,
    "cy": 638.8
}

def colorize_point_cloud(lidar_points, image, transformed_points):
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



if __name__=="__main__":
    # Paths to ROS bags and output file
    lidar_bag_path = "/home/knadmin/Ashwin/AURORA_dataset/Segmented3DMap/30_bridgecurve_80m_100kmph_BTUwDLR.bag"
    odometry_bag_path = "/home/knadmin/Ashwin/AURORA_dataset/Segmented3DMap/30_bridgecurve_80m_100kmph_BTUwDLR_trajectory.bag"
    output_pcd_file = "30_bridgecurve_80m_100kmph_BTUwDLR_trajectory_output_map_filtered.ply"
    output_trajectory_pcd_file = "30_bridgecurve_80m_100kmph_BTUwDLR_trajectory_output_map_trajectory.ply"
    lidar_topic = "/VLP32/velodyne_points"
    image_topic = "/zed2i/zed_node/left/image_rect_color/compressed"
    odometry_topic = "/lio_sam/mapping/odometry"
    output_txt_dir = "30_bridgecurve_80m_100kmph_BTUwDLR"

    trajectory_flag = True
    apply_ransac_flag = True
    compute_map_entropy_flag = True
    downsample_pcd_flag = True

    frame_count = 0  # Counter for frame-based naming

    if os.path.exists(output_txt_dir):
        shutil.rmtree(output_txt_dir)  
    os.makedirs(output_txt_dir, exist_ok=True)  # Create new empty folder

    cv2.namedWindow("Display_Image", cv2.WINDOW_NORMAL) 
    cv2.namedWindow("Display_Image2", cv2.WINDOW_NORMAL) 


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
    colors_list = []
    intensity_list = []
    ring_list = []

    lidar_bag = rosbag.Bag(lidar_bag_path, "r")
    image = None  # Initialize image storage



    points_counter =0

    for topic, msg, t in lidar_bag.read_messages(topics=[lidar_topic, image_topic]):
        timestamp = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9  # High-precision timestamp
        
        if topic == image_topic:
            image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            continue  # Store the latest image and move on
        
        if topic == lidar_topic and image is not None:
            # Find the closest odometry timestamp
            closest_idx = np.argmin(np.abs(np.array(odometry_timestamps) - timestamp))
            closest_time, position, orientation = odometry_data[closest_idx]

            print("**************************")
            print("Frame counter", frame_count)
            
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
            valid_points = (x_proj >= 0) & (x_proj < image.shape[1]) & (y_proj >= 0) & (y_proj < image.shape[0])
            count_trues_projected = np.sum(valid_points)
            print("Number of valid points after using points within image {} (and after z axes filtering)".format(count_trues_projected))

            x_proj, y_proj = x_proj[valid_points], y_proj[valid_points]

            # 1️⃣ Create the original image with red-projected points
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            image_with_red = image.copy()
            point_size = 2
            for x, y in zip(x_proj, y_proj):
                x_start, x_end = max(0, x - point_size), min(image.shape[1], x + point_size + 1)
                y_start, y_end = max(0, y - point_size), min(image.shape[0], y + point_size + 1)
                image_with_red[y_start:y_end, x_start:x_end] = [0, 0, 255]  # Red color
                mask[y_start:y_end, x_start:x_end] = 255  # White mask

            colored_points_rgb, valid_indices  = colorize_point_cloud(lidar_points, image, transformed_points)

            #Part of the code to use for using semantic iamge color
            # colored_points_semantic, valid_indices  = colorize_point_cloud(lidar_points, semantic_image, transformed_points)



            # Save transformed + colorized LiDAR points to TXT
            save_pointcloud_to_txt(frame_count, transformed_points, colored_points_rgb, valid_indices, timestamp, lidar_points)

            frame_count += 1  # Increment frame count
            
            valid_points = points[valid_indices] 
        

            # colors = np.zeros((points.shape[0], 3), dtype=np.uint8)

            print("checking length", len(colored_points_rgb), len(valid_indices))
            points_counter += len(valid_indices)
            # import pdb;pdb.set_trace()

            black_image = overlay_points_on_black_image(x_proj, y_proj, image)


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
            colors = np.zeros((len(valid_indices), 3))  # Initialize color array

            total_points = len(valid_indices)  # Total number of valid colorized points
            non_black_count = 0  # Counter for non-black points

            for idx, point in enumerate(colored_points_rgb):
                x, y, z, r, g, b = point  # Unpack each point with color values

                # Normalize RGB values (Open3D expects colors in range [0,1])
                r_norm = r / 255.0
                g_norm = g / 255.0
                b_norm = b / 255.0

                colors[idx] = [r_norm, g_norm, b_norm]
                # colors[idx] = [1, 0, 0]

                # Check if the point is non-black
                if (r, g, b) != (0, 0, 0):
                    non_black_count += 1

                # Debugging: Print first 10 points
                # if idx < 10:
                #     print(f"Point {idx}: XYZ=({x:.3f}, {y:.3f}, {z:.3f}) RGB=({r}, {g}, {b}) -> Normalized=({r_norm:.3f}, {g_norm:.3f}, {b_norm:.3f})")

            # Final Debugging Check
            # print(f"Out of {total_points} colorized points, {non_black_count} have non-black colors.")
            # if non_black_count == 0:
            #     print("⚠ Warning: All colorized points are black! Check projection and color extraction logic.")

        
            # Append to global lists
            pcd_points.append(transformed_points)
            colors = colors.astype(np.float32)

            colors_list.append(colors)
            intensity_list.append(intensity)
            ring_list.append(ring)

            cv2.imshow("Display_Image", image_with_red) 
            cv2.imshow("Display_Image2", black_image) 
            cv2.waitKey(50) 


    cv2.destroyAllWindows()
    print("************************")

    # Convert to Open3D format
    pcd_points = np.vstack(pcd_points)
    colors_list = np.vstack(colors_list)
    intensity_list = np.hstack(intensity_list)[:, None]  # Ensure correct shape
    ring_list = np.hstack(ring_list)[:, None]

    print(f"Total points saved in final PCD: {pcd_points.shape[0]}")
    print(f"Total points from valid indexes across entire rosbag: {points_counter}")


    # Create structured Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    pcd.colors = o3d.utility.Vector3dVector(colors_list)




    if apply_ransac_flag:
        # Apply RANSAC to remove water reflections (ground-like surface)
        pcd = remove_water_using_ransac(pcd)

    if compute_map_entropy_flag:
        # Compute entropy
        entropy, nearest_distances = compute_map_entropy(pcd)
        plot_entropy_distribution(nearest_distances)

    if downsample_pcd_flag:
        pcd = voxel_grid_downsample(pcd, voxel_size=0.2)  # Adjust voxel size for detail

    if trajectory_flag:
        # Generate trajectory point cloud for the boat's movement
        trajectory_pcd = generate_boat_trajectory_pcd(odometry_data)

        # Merge LiDAR and trajectory point clouds into one visualization
        combined_pcd = pcd + trajectory_pcd  # Combine both datasets


    end_time = time.time()
    time_taken = end_time - start_time
    print("Total TIme taken",time_taken)

    # Save the final point cloud
    print(f"Saving final 3D map to {output_pcd_file}")
       

    if trajectory_flag:

        o3d.io.write_point_cloud(output_pcd_file, combined_pcd)
        o3d.io.write_point_cloud(output_trajectory_pcd_file, trajectory_pcd)
        # Visualize both LiDAR map and Boat trajectory together
        o3d.visualization.draw_geometries([pcd, trajectory_pcd], 
                                        window_name="3D Map + Boat Trajectory",
                                        point_show_normal=False)
        o3d.visualization.draw_geometries([trajectory_pcd], 
                                        window_name="Boat Trajectory",
                                        point_show_normal=False)
    else:
        o3d.io.write_point_cloud(output_pcd_file, pcd)
        # Visualize the final point cloud
        o3d.visualization.draw_geometries([pcd], window_name="Photorealistic 3D Map (Not yet)",
                                        point_show_normal=False)

    lidar_bag.close()








    #2432116



        




