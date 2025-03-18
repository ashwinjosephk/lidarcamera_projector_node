import open3d as o3d
import numpy as np
import time
import glob
import yaml
import re

ROOT_DIR = "../../python_pcd/"
DATA_DIR = "DATA/30_bridgecurve_80m_100kmph_BTUwDLR/output_pcd/"

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
    The .txt file has the following structure: msg timestamp, x, y, z, intensity, ring, time

    Args:
        txt_path (str): path of the .txt file (e.g., "/path/pcd_01.txt").
    """
    try:
        data = np.loadtxt(txt_path, delimiter=",", usecols=(1, 2, 3, 7, 8, 9))  # Read only x, y, z columns

        points = np. column_stack((data[:, 0], data[:, 1], data[:, 2]))
        colors = data[:, 3:6] / 255.0

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        return cloud
    except Exception as e:
        raise RuntimeError(f"Failed to read .txt file at {txt_path}. Error: {str(e)}")
    
def read_txt_file_ashwin(txt_path):
    """
    Reads a .txt file and returns an Open3D point cloud.
    The .txt file has the following structure: msg timestamp, x, y, z, intensity, ring, time

    Args:
        txt_path (str): path of the .txt file (e.g., "/path/pcd_01.txt").
    """
    try:
        data = np.loadtxt(txt_path, delimiter=",", usecols=(0, 1, 2, 6, 7, 8))  # Read only x, y, z columns

        points = np. column_stack((data[:, 0], data[:, 1], data[:, 2]))
        colors = data[:, 3:6] / 255.0

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        return cloud
    except Exception as e:
        raise RuntimeError(f"Failed to read .txt file at {txt_path}. Error: {str(e)}")

def extract_number(filename):
    """Extracts the last numeric part of a filename before the extension."""
    match = re.search(r'(\d+)(?=\.\w+$)', filename)  # Find the last number before file extension
    return int(match.group(1)) if match else float('inf')  # Use inf if no match (to push invalid names last)

def visualize_multiple_pointclouds(file_pattern, frame_delay=0.1):
    """
    Visualizes multiple point clouds in sequence (animation effect).

    Args:
        file_pattern (str): Pattern to find .txt files (e.g., "/path/*.txt").
        frame_delay (float): Delay between frames for animation.
    """
    # Find all point cloud files matching the pattern and sort them numerically
    pointcloud_files = sorted(glob.glob(file_pattern), key=lambda x: extract_number(x.split("/")[-1]))

    if not pointcloud_files:
        print("No point cloud files found!")
        return

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

    # Load the first point cloud
    cloud = read_txt_file(pointcloud_files[0])
    # cloud = read_txt_file_ashwin(pointcloud_files[0])
    vis.add_geometry(cloud)

    print(f"Displaying {len(pointcloud_files)} point clouds...")

    for file in pointcloud_files:
        new_cloud = read_txt_file(file)  # Load next point cloud
        # new_cloud = read_txt_file_ashwin(file)  # Load next point cloud
        cloud.points = new_cloud.points  # Update points in existing cloud
        cloud.colors = new_cloud.colors  # Update points in existing cloud

        vis.update_geometry(cloud)  # Update the displayed cloud
        vis.poll_events()  # Process visualization events
        vis.update_renderer()  # Render the updated scene

        time.sleep(frame_delay)  # Small delay for smooth animation

    vis.run()  # Keep the window open until user closes it
    vis.destroy_window()

if __name__ == "__main__":
    try:
        # Run visualization
        config_path = ROOT_DIR + "config/pcd_config.yaml"
        config = load_config(config_path)

        file_pattern = DATA_DIR + config["pcd_files_list_path"]  
        visualize_multiple_pointclouds(file_pattern, frame_delay=0.1)
    except Exception as e:
        print(f"Error: {str(e)}")



