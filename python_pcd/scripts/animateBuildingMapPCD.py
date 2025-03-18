from nav_msgs.msg import Odometry
import open3d as o3d
import numpy as np
import yaml
import glob
import time
import sys
import os

ROOT_DIR = "../../python_pcd/"
DATA_DIR = "../../DATA/30_bridgecurve_80m_100kmph_BTUwDLR/output_pcd/"

def load_config(config_path):
    """Load the YAML configuration."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration file: {config_path}. Error: {str(e)}")
    return config

def read_txt_file(txt_path):
    """
    Reads a .txt file and returns an Open3D point cloud.

    Args:
        txt_path (str): Path to the .txt file.

    Returns:
        o3d.geometry.PointCloud: The loaded point cloud.
    """
    try:
        # Load text file
        data = np.loadtxt(txt_path, delimiter=",", usecols=(1, 2, 3, 7, 8, 9))  # Read only x, y, z columns

        points = np.column_stack((data[:, 0], data[:, 1], data[:, 2]))
        # mask = ~(
        #     ((points[:, 0] <= 4.0) & (points[:, 0] >= -10.0) & 
        #      (points[:, 1] <= 2.0) & (points[:, 1] >= -2.0) & 
        #      (points[:, 2] <= 0.6) & (points[:, 2] >= -2.0)))
        # filtered_points = points[mask]

        colors = data[:, 3:6] / 255.0
        
        # Create Open3D point cloud
        cloud = o3d.geometry.PointCloud()
        # cloud.points = o3d.utility.Vector3dVector(filtered_points)
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        
        return cloud
    except Exception as e:
        raise RuntimeError(f"Failed to read .txt file at {txt_path}. Error: {str(e)}") 

def read_bin_file(bin_path):
    """
    Reads a binary .bin file and returns an Open3D point cloud.

    Args:
        bin_path (str): Path to the .bin file.

    Returns:
        o3d.geometry.PointCloud: The loaded point cloud.
    """
    try:
        # Load binary file
        data = np.fromfile(bin_path, dtype=np.float32)
        
        # Reshape into Nx4 (x, y, z, intensity) or Nx3 (x, y, z)
        points = data.reshape((-1, 4))[:, :3]  # Only take x, y, z columns
        
        # Create Open3D point cloud
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)

        return cloud
    except Exception as e:
        raise RuntimeError(f"Failed to read .bin file at {bin_path}. Error: {str(e)}")

def read_trajectory(trajectory_file):
    """Reads the trajectory file and extracts timestamps, positions, and orientations."""
    data = np.loadtxt(trajectory_file, delimiter=",")
    
    timestamps = data[:, 0]  # Assuming time is in the 8th column (index 7)
    positions = data[:, 1:4]  # Columns 4-6 (x, y, z)
    # quaternions = data[:, 5:]  # Columns 0-3 (w, x, y, z)
    quaternions = np.column_stack((data[:, 7], data[:, 4], data[:, 5], data[:, 6]))
    
    return timestamps, quaternions, positions

def print_progress_bar(iteration, total, length=60):
    """
    Print a progress bar in the console.
    
    Args:
        iteration (int): Current iteration.
        total (int): Total iterations.
        length (int): Length of the progress bar.
    """
    percent = (iteration / total) * 100
    filled_length = int(length * iteration // total)
    bar = '=' * filled_length + '.' * (length - filled_length)
    sys.stdout.write(f"\r  Building Map: [{bar}] {percent:.1f}%")
    sys.stdout.flush()

def create_rectangle(min_corner, max_corner, color=[1.0, 1.0, 1.0]):
    """
    Create a rectangle in the XY plane (horizontal plane).

    Args:
        min_corner (list): Bottom-left corner of the rectangle.
        max_corner (list): Top-right corner of the rectangle.
        color (list): RGB color for the rectangle.
    """
    rectangle = o3d.geometry.TriangleMesh()

    # Define the four corners of the rectangle
    bottom_left = np.array(min_corner)
    bottom_right = np.array([max_corner[0], min_corner[1], min_corner[2]])
    top_left = np.array([min_corner[0], max_corner[1], min_corner[2]])
    top_right = np.array(max_corner)

    # Add vertices
    rectangle.vertices = o3d.utility.Vector3dVector(
        [bottom_left, bottom_right, top_left, top_right]
    )

    # Add triangles (two triangles to form a rectangle)
    rectangle.triangles = o3d.utility.Vector3iVector(
        [[0, 1, 2], [2, 1, 3]]
    )

    # Add color for each vertex
    rectangle.vertex_colors = o3d.utility.Vector3dVector([color] * 4)

    return rectangle

def visualize_mapcloud(cloud, config):
    """Visualize the point cloud and optional rectangle."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Map cloud",
        width=config["window_width"],
        height=config["window_height"],
    )

    opt = vis.get_render_option()
    opt.point_size = config["point_size"]
    opt.background_color = np.array(config["background_color"])
    
    # Point color option
    color_option = config.get("point_color_option", "Default")
    if color_option == "Default":
        opt.point_color_option = o3d.visualization.PointColorOption.Default
    elif color_option == "Color":
        opt.point_color_option = o3d.visualization.PointColorOption.Color
    elif color_option == "Normal":
        opt.point_color_option = o3d.visualization.PointColorOption.Normal
    elif color_option == "XCoordinate":
        opt.point_color_option = o3d.visualization.PointColorOption.XCoordinate
    elif color_option == "YCoordinate":
        opt.point_color_option = o3d.visualization.PointColorOption.YCoordinate
    elif color_option == "ZCoordinate":
        opt.point_color_option = o3d.visualization.PointColorOption.ZCoordinate
    else:
        print(f"Invalid point_color_option: {color_option}. Using Default option.")
        opt.point_color_option = o3d.visualization.PointColorOption.Default

    # Show coordinate frame if specified
    opt.show_coordinate_frame = config.get("show_coordinate_frame", False)

    vis.add_geometry(cloud)

    # Add horizontal rectangle if specified
    if config.get("add_horizontal_plane", False):
        min_bound = np.asarray(cloud.get_min_bound())
        max_bound = np.asarray(cloud.get_max_bound())

        bottom_left = [
            min_bound[0],
            min_bound[1],
            config["rectangle"]["bottom_left"][2],
        ]
        top_right = [
            max_bound[0],
            max_bound[1],
            config["rectangle"]["top_right"][2],
        ]
        color = config["rectangle"]["color"]

        rectangle = create_rectangle(bottom_left, top_right, color)
        vis.add_geometry(rectangle)

    vis.run()
    vis.destroy_window()

def quaternion_to_rotation_matrix(q):
    """Converts a quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = q
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])
    return R

def animate_map_building_and_save_image_frames(file_pattern, trajectory_file, frame_delay, save_frames):
    """Animates the process of merging multiple point clouds into one global map and records the animation."""
    
    # Read trajectory data
    timestamps, quaternions, positions = read_trajectory(trajectory_file)

    # Get sorted point cloud files
    pointcloud_files = sorted(glob.glob(file_pattern), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    # import pdb;pdb.set_trace()
    if len(pointcloud_files) != len(positions):
        raise ValueError("Mismatch between number of point clouds and trajectory entries!")

    # Create output directory for frames
    frame_dir = DATA_DIR + config["frames_out"]
    if save_frames:
        os.makedirs(frame_dir, exist_ok=True)

    # Initialize Open3D Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Map cloud",
        width=config["window_width"],
        height=config["window_height"],
    )

    opt = vis.get_render_option()
    opt.point_size = config["point_size"]
    opt.background_color = np.array(config["background_color"])

    merged_cloud = o3d.geometry.PointCloud()
    total_files = len(pointcloud_files)

    for i, file in enumerate(pointcloud_files):
        print_progress_bar(i, total_files)

        # Read the point cloud
        cloud = read_txt_file(file)

        # Convert quaternion to rotation matrix
        R = quaternion_to_rotation_matrix(quaternions[i])

        # Get the translation (position)
        t = positions[i]

        # Apply transformation (Rotation + Translation)
        points = np.asarray(cloud.points)
        transformed_points = (R @ points.T).T + t  # Apply rotation & translation

        # Update the cloud points
        cloud.points = o3d.utility.Vector3dVector(transformed_points)

        # Merge with the global point cloud
        merged_cloud += cloud

        # Update visualization
        if i == 0:
            vis.add_geometry(merged_cloud)  # Add first cloud
        else:
            vis.update_geometry(merged_cloud)  # Update existing cloud
        
        vis.poll_events()  # Process Open3D events
        vis.update_renderer()  # Refresh visualization

        # Capture and save frame
        if save_frames:
            image_path = os.path.join(frame_dir, f"frame_{i:04d}.png")
            vis.capture_screen_image(image_path)

        time.sleep(frame_delay)  # Small delay for smooth animation

    print("Map building animation complete. Saving final merged map...")
    
    # Save the final merged map
    o3d.io.write_point_cloud(DATA_DIR + config["pcd_out"], merged_cloud)

    # Keep visualization open
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    try:
        # Load YAML configuration
        config_path = ROOT_DIR + "config/pcd_config.yaml"
        config = load_config(config_path)

        frame_delay = 0.1
        save_frames = config["save_frames"]
        
        # Load all the pointcloud files
        pointcloud_files_path = DATA_DIR + config["pcd_in"]

        # Load the trajectory information
        trajectory_path = DATA_DIR + config["trajectory_in"]

        animate_map_building_and_save_image_frames(pointcloud_files_path, trajectory_path, frame_delay, save_frames)
    except Exception as e:
        print(f"Error: {str(e)}")

