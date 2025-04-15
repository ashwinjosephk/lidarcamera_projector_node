import open3d as o3d
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import heapq
import matplotlib.patches as mpatches

CATEGORY_COLORS = {
    0: ([135, 206, 250], "Sky"),
    1: ([0, 191, 255], "Water"),
    2: ([50, 205, 50], "Vegetation"),
    3: ([200, 0, 0], "Riverbank"),
    4: ([184, 134, 11], "Bridge"),
    5: ([157, 0, 255], "Other")
}

def get_color_by_name(category_name, CATEGORY_COLORS):
    for color, name in CATEGORY_COLORS.values():
        if name == category_name:
            return color
    return None

def build_occupancy_grid_map(pcd_semantic, output_folder, lidar_bag_base_name, free_navigatable_color,kernel):
    resolution = 0.2
    z_min, z_max = -1.0, 3.0
    water_rgb = np.array(free_navigatable_color) / 255.0
    color_tolerance = 0.05
    min_cluster_area = 5

    points = np.asarray(pcd_semantic.points)
    colors = np.asarray(pcd_semantic.colors)
    valid_z = (points[:, 2] > z_min) & (points[:, 2] < z_max)
    points = points[valid_z]
    colors = colors[valid_z]

    is_water = np.linalg.norm(colors - water_rgb, axis=1) < color_tolerance

    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    width = int(np.ceil((max_x - min_x) / resolution))
    height = int(np.ceil((max_y - min_y) / resolution))

    occupancy_grid = np.full((height, width), 127, dtype=np.uint8)
    for pt in points[is_water]:
        grid_x = int((pt[0] - min_x) / resolution)
        grid_y = int((pt[1] - min_y) / resolution)
        if 0 <= grid_x < width and 0 <= grid_y < height:
            occupancy_grid[grid_y, grid_x] = 0
    for pt in points[~is_water]:
        grid_x = int((pt[0] - min_x) / resolution)
        grid_y = int((pt[1] - min_y) / resolution)
        if 0 <= grid_x < width and 0 <= grid_y < height:
            occupancy_grid[grid_y, grid_x] = 255

    occupancy_grid_flipped = np.flipud(occupancy_grid)
    _, binary = cv2.threshold(occupancy_grid_flipped, 254, 255, cv2.THRESH_BINARY)
    
    binary = cv2.dilate(binary, kernel, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    clustered_grid = np.zeros_like(binary, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_cluster_area:
            clustered_grid[labels == i] = 255

    contours, _ = cv2.findContours(clustered_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_clusters = np.zeros_like(clustered_grid)
    cv2.drawContours(filled_clusters, contours, -1, color=255, thickness=-1)

    if len(contours) >= 2:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        shore1 = contours[0].squeeze()
        shore2 = contours[1].squeeze()
        shore1_sorted = shore1[np.argsort(shore1[:, 1])]
        shore2_sorted = shore2[np.argsort(shore2[:, 1])]

        waterway_polygon = np.vstack([shore1_sorted, shore2_sorted[::-1]]).astype(np.int32)
        h, w = filled_clusters.shape
        color_map = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.fillPoly(color_map, [waterway_polygon], color=(255, 0, 0))  # blue
        color_map[filled_clusters == 255] = [0, 0, 255]
        mask_total = (np.any(color_map != [0, 0, 0], axis=2)).astype(np.uint8)
        color_map[mask_total == 0] = [0, 255, 0]  # purple
    else:
        color_map = cv2.cvtColor(filled_clusters, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(f"{output_folder}/{lidar_bag_base_name}_semantic_occupancy_grid.png", occupancy_grid_flipped)
    cv2.imwrite(f"{output_folder}/{lidar_bag_base_name}_semantic_occupancy_clusters.png", clustered_grid)
    cv2.imwrite(f"{output_folder}/{lidar_bag_base_name}_semantic_occupancy_filled_clusters.png", filled_clusters)
    final_map_path = f"{output_folder}/{lidar_bag_base_name}_explicit_waterway_map.png"
    cv2.imwrite(final_map_path, color_map)
    print(f"✅ Final waterway corridor map saved: {final_map_path}")

    # cv2.namedWindow("Final Semantic Waterway Map", cv2.WINDOW_NORMAL)
    # cv2.imshow("Final Semantic Waterway Map", color_map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    # Visualization using matplotlib with grid and coordinates
    # Visualization using matplotlib with grid and coordinates
    grid_square_size = 10
    x_ticks = np.arange(0, width + 1, grid_square_size)
    y_ticks = np.arange(0, height + 1, grid_square_size)

    fig, ax = plt.subplots(figsize=(12, 12 * (height / width)))
    ax.imshow(color_map)
    ax.set_title("Semantic Occupancy Grid with Grid Overlay")
    ax.set_xlabel("X (grid cells)")
    ax.set_ylabel("Y (grid cells)")

    # Set all ticks for the grid
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Only show every Nth tick label on X-axis
    label_every = 10  # Show every second label
    ax.set_xticklabels(
        [str(tick) if i % label_every == 0 else '' for i, tick in enumerate(x_ticks)]
    )
    ax.set_yticklabels(
        [str(tick) if i % label_every == 0 else '' for i, tick in enumerate(y_ticks)]
    )

    # Draw the grid
    ax.grid(color='gray', linestyle='--', linewidth=0.5)

    ax.set_aspect('equal', adjustable='box')
    plt.axis('scaled')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Create a refined binary map based on red presence in each grid square
    refined_map = np.zeros_like(color_map)  # Same size, default black

    for y in range(0, height, grid_square_size):
        for x in range(0, width, grid_square_size):
            # Define grid cell
            y_end = min(y + grid_square_size, height)
            x_end = min(x + grid_square_size, width)

            cell = color_map[y:y_end, x:x_end]



            # Check if there's any red-ish pixel (you can adjust threshold)
            # Check for red-ish pixels
            red_mask = (cell[:, :, 0] > 150) & (cell[:, :, 1] < 100) & (cell[:, :, 2] < 100)

            # Check for green-ish pixels
            green_mask = (cell[:, :, 1] > 150) & (cell[:, :, 0] < 100) & (cell[:, :, 2] < 100)

            # If any red or green is present, mark as not navigable
            if np.any(red_mask | green_mask):
                refined_map[y:y_end, x:x_end] = [255, 0, 0]  # Red for obstacle
            else:
                refined_map[y:y_end, x:x_end] = [0, 0, 255]  # Blue for safe

    # Plot the refined map
    fig, ax = plt.subplots(figsize=(12, 12 * (height / width)))
    ax.imshow(refined_map)
    ax.set_title("Refined Navigability Map (Red = Blocked, Blue = Safe)")
    ax.set_xlabel("X (grid cells)")
    ax.set_ylabel("Y (grid cells)")
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(
        [str(tick) if i % label_every == 0 else '' for i, tick in enumerate(x_ticks)]
    )
    ax.set_yticklabels(
        [str(tick) if i % label_every == 0 else '' for i, tick in enumerate(y_ticks)]
    )
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('scaled')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Convert RGB to BGR for OpenCV display
    original_bgr = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
    refined_bgr = cv2.cvtColor(refined_map, cv2.COLOR_RGB2BGR)

    # Stack images horizontally (side by side)
    combined = np.hstack((original_bgr, refined_bgr))

    # Display in OpenCV window
    cv2.namedWindow("Original vs Refined Map", cv2.WINDOW_NORMAL)
    cv2.imshow("Original vs Refined Map", combined)
    cv2.waitKey(50)
    cv2.destroyAllWindows()
    # Alpha blend original and refined maps
    alpha = 0.6  # Original image weight
    beta = (1 - alpha)  # Refined mask weight

    # Convert both to BGR for OpenCV
    original_bgr = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
    
    refined_bgr = cv2.cvtColor(refined_map, cv2.COLOR_RGB2BGR)

    # Make sure they are the same size (in case)
    # assert original_bgr.shape == refined_bgr.shape, "Image sizes don't match!"

    # Blend the two images
    blended = cv2.addWeighted(original_bgr, alpha, refined_bgr, beta, 0)

    # Display the blended result
    # cv2.namedWindow("Blended Map Overlay", cv2.WINDOW_NORMAL)
    # cv2.imshow("Blended Map Overlay", blended)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Convert blended image from BGR to RGB for matplotlib
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    # Plot using matplotlib
    fig, ax = plt.subplots(figsize=(12, 12 * (height / width)))
    ax.imshow(blended_rgb)
    ax.set_title("Blended Overlay Map (Original + Refined Obstacle Zones)")
    ax.set_xlabel("X (grid cells)")
    ax.set_ylabel("Y (grid cells)")
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(
        [str(tick) if i % label_every == 0 else '' for i, tick in enumerate(x_ticks)]
    )
    ax.set_yticklabels(
        [str(tick) if i % label_every == 0 else '' for i, tick in enumerate(y_ticks)]
    )
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('scaled')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

    import heapq

    # Step 1: Convert refined_map into binary grid (0 = obstacle, 1 = free)
    grid_rows = height // grid_square_size
    grid_cols = width // grid_square_size
    grid = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

    for y in range(grid_rows):
        for x in range(grid_cols):
            y_start = y * grid_square_size
            x_start = x * grid_square_size
            cell = refined_map[y_start:y_start + grid_square_size, x_start:x_start + grid_square_size]
            is_obstacle = np.all(cell == [255, 0, 0])
            grid[y, x] = 0 if is_obstacle else 1



    # Step 3: A* Pathfinding
    # def heuristic(a, b):
    #     return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def heuristic(a, b):
        dx = abs(a[1] - b[1])
        dy = abs(a[0] - b[0])
        return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)


    def a_star(grid, start, goal):
        rows, cols = grid.shape
        open_set = []
        heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            # for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:

                ny, nx = current[0] + dy, current[1] + dx
                neighbor = (ny, nx)
                if 0 <= ny < rows and 0 <= nx < cols and grid[ny, nx] == 1:
                    # tentative_g = current_g + 1
                    if dy != 0 and dx != 0:
                        move_cost = 1.414  # √2 for diagonal
                    else:
                        move_cost = 1.0
                    tentative_g = current_g + move_cost

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f = tentative_g + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f, tentative_g, neighbor))
        return None

    # Manually specified start and goal (in pixel coordinates)
    start_px = (200, 200)
    goal_px = (1160, 400)

    # Convert to grid coordinates
    start = (start_px[1] // grid_square_size, start_px[0] // grid_square_size)
    goal = (goal_px[1] // grid_square_size, goal_px[0] // grid_square_size)

    print(f"Converted Start (grid): {start}, Goal (grid): {goal}")
    print(f"Grid shape: {grid.shape}")

    # Sanity checks
    if not (0 <= start[0] < grid.shape[0] and 0 <= start[1] < grid.shape[1]):
        print("❌ Start point is out of bounds.")
        path = None
    elif not (0 <= goal[0] < grid.shape[0] and 0 <= goal[1] < grid.shape[1]):
        print("❌ Goal point is out of bounds.")
        path = None
    elif grid[start[0], start[1]] == 0:
        print("❌ Start point is in an obstacle.")
        path = None
    elif grid[goal[0], goal[1]] == 0:
        print("❌ Goal point is in an obstacle.")
        path = None
    else:
        print("✅ Both points are valid, starting A* pathfinding...")
        path = a_star(grid, start, goal)


    # Step 4: Visualize the path on the blended RGB map
    fig, ax = plt.subplots(figsize=(12, 12 * (height / width)))
    original_bgr = cv2.cvtColor(original_bgr, cv2.COLOR_RGB2BGR)
    ax.imshow(original_bgr)
    ax.set_title("A* Path over Blended Map")
    ax.set_xlabel("X (grid cells)")
    ax.set_ylabel("Y (grid cells)")




    # Create legend handles
    legend_elements = [
        mpatches.Patch(color='blue', label='Navigable'),
        mpatches.Patch(color='red', label='Riverbank Obstacle'),
        mpatches.Patch(color='orange', label='A* Path'),
        mpatches.Patch(color='purple', label='Start Point'),
        mpatches.Patch(color='darkred', label='Goal Point'),
        mpatches.Patch(color='lightgreen', label='Land Obstacle')
    ]

    # Add the legend to the plot
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=True)


    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(
        [str(tick) if i % label_every == 0 else '' for i, tick in enumerate(x_ticks)]
    )
    ax.set_yticklabels(
        [str(tick) if i % label_every == 0 else '' for i, tick in enumerate(y_ticks)]
    )
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('scaled')
    ax.invert_yaxis()

    if path:
        for (gy, gx) in path:
            rect = plt.Rectangle(
                (gx * grid_square_size, gy * grid_square_size),
                grid_square_size, grid_square_size,
                linewidth=0, edgecolor='none', facecolor='orange'
            )

            ax.add_patch(rect)
    else:
        print("⚠️ No path found between start and goal.")
    # Highlight the start point (e.g., green)
    start_rect = plt.Rectangle(
        (start[1] * grid_square_size, start[0] * grid_square_size),
        grid_square_size, grid_square_size,
        linewidth=1, edgecolor='black', facecolor='purple'
    )
    ax.add_patch(start_rect)

    # Highlight the goal point (e.g., red)
    goal_rect = plt.Rectangle(
        (goal[1] * grid_square_size, goal[0] * grid_square_size),
        grid_square_size, grid_square_size,
        linewidth=1, edgecolor='black', facecolor='darkred'
    )
    ax.add_patch(goal_rect)
    plt.tight_layout()
    plt.show()




# Paths
    lidar_bags_base_name_list = ['7_anlegen_80m_100kmph_BTUwDLR', '9_anlegen_80m_100kmph_BTUwDLR', '17_straight_200m_100kmph_BTUwDLR', '29_bridgecurve_80m_100kmph_BTUwDLR','30_bridgecurve_80m_100kmph_BTUwDLR','37_curvepromenade_160m_100kmph_BTUwDLR','53_schleuseeinfahrt_20m_100kmph_BTUwDLR','2023.11.07_friedrichstrasseToBerlinDom']

# lidar_bag_base_name = "37_curvepromenade_160m_100kmph_BTUwDLR"
# lidar_bag_base_name = "9_anlegen_80m_100kmph_BTUwDLR"
# lidar_bag_base_name = "53_schleuseeinfahrt_20m_100kmph_BTUwDLR" #maybe
lidar_bag_base_name = "37_curvepromenade_160m_100kmph_BTUwDLR"

output_txt_dir = f"/home/knadmin/Ashwin/AURORA_dataset/Segmented3DMap/DATA/{lidar_bag_base_name}"
output_folder = os.path.join(output_txt_dir, "output_pcd")
output_pcd_semantic_file = f"{output_folder}/{lidar_bag_base_name}_trajectory_output_map_semantic.ply"


# kernel = np.ones((3, 3), np.uint8)
kernel = np.ones((5, 5), np.uint8)


pcd_semantic = o3d.io.read_point_cloud(output_pcd_semantic_file)
print(f"✅ Loaded {len(pcd_semantic.points)} points from saved semantic PCD")

free_navigatable_color = get_color_by_name("Water", CATEGORY_COLORS)
if free_navigatable_color is None:
    raise ValueError("Could not find color for class 'Water'")

build_occupancy_grid_map(pcd_semantic, output_folder, lidar_bag_base_name, free_navigatable_color, kernel)
