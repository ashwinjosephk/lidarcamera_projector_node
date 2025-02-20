import open3d as o3d
import yaml
import numpy as np

ROOT_DIR = "/home/knadmin/Ashwin/AURORA_dataset/Segmented3DMap/finevisualize/"

def load_config(config_path):
    """Load the YAML configuration."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration file: {config_path}. Error: {str(e)}")
    return config


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
    opt.point_color_option = o3d.visualization.PointColorOption.Color
    print("Color option:",color_option)

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


def print_pointcloud_info(cloud):
    """Print point cloud details."""
    print(f"Number of points: {len(cloud.points)}")

    if len(cloud.points) > 0:
        for i in range(min(5, len(cloud.points))):
            point = cloud.points[i]
            print(f"Point {i}: {point}")

    if cloud.has_colors():
        print("The point cloud contains colors. Sample color data:")
        for i in range(min(5, len(cloud.colors))):
            color = cloud.colors[i]
            print(f"Color {i}: {color}")

    if cloud.has_normals():
        print("The point cloud contains normals. Sample normal data:")
        for i in range(min(5, len(cloud.normals))):
            normal = cloud.normals[i]
            print(f"Normal {i}: {normal}")


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


if __name__ == "__main__":
    try:
        # Load YAML configuration
        config_path = ROOT_DIR + "config/pcd_config.yaml"
        config = load_config(config_path)

        # Determine the file type and load the point cloud
        pcd_path = ROOT_DIR + config["pcd_file_path"]
        if pcd_path.endswith(".pcd"):
            map_cloud = o3d.io.read_point_cloud(pcd_path)
        elif pcd_path.endswith(".bin"):
            map_cloud = read_bin_file(pcd_path)
        elif pcd_path.endswith(".ply"):
            map_cloud = o3d.io.read_point_cloud(pcd_path)  # Add this line to handle .ply
        else:
            raise ValueError(f"Unsupported file format: {pcd_path}")

        # Visualize and print info
        visualize_mapcloud(map_cloud, config)
        print_pointcloud_info(map_cloud)
    except Exception as e:
        print(f"Error: {str(e)}")

