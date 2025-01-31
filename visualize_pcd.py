import open3d as o3d

# Load PCD file
pcd = o3d.io.read_point_cloud("/home/knadmin/Ashwin/AURORA_dataset/Segmented3DMap/30_bridgecurve_80m_100kmph_BTUwDLR_trajectory_output_map_filtered.ply")

# Print basic information
print(pcd)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd], window_name="PCD Viewer")

