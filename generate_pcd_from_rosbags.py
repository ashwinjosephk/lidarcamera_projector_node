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

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch
from PIL import Image
import yaml
from viewPcdO3d import visualize_mapcloud, create_rectangle, filter_pcd_by_class, voxel_grid_downsample, visualize_pcd_with_custom_settings
from stereodepth import compute_depth_from_stereo
from utils import save_pointcloud_to_txt, compute_map_entropy, plot_entropy_distribution, filter_points_by_class, apply_color_map, generate_boat_trajectory_pcd, remove_water_using_ransac, load_config, load_camera_intrinsics, overlay_points_on_black_image, load_category_colors, colorize_point_cloud_semantic, colorize_point_cloud_photorealisitc
import threading
from queue import Queue



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


def combine_visualization(vis_semantic, image_with_red, vis_photo, overlayed_image): 
    """
    Combines OpenCV images and Open3D visualization into a single window.
    Ensures no image distortion while saving video.
    """
    vis_semantic.poll_events()
    vis_semantic.update_renderer()
    vis_photo.poll_events()
    vis_photo.update_renderer()
    
    # Capture Open3D renderings **exactly as displayed**
    render_semantic = vis_semantic.capture_screen_float_buffer(do_render=True)
    render_semantic_np = (np.asarray(render_semantic) * 255).astype(np.uint8)
    render_semantic_np = cv2.cvtColor(render_semantic_np, cv2.COLOR_RGB2BGR)

    render_photo = vis_photo.capture_screen_float_buffer(do_render=True)
    render_photo_np = (np.asarray(render_photo) * 255).astype(np.uint8)
    render_photo_np = cv2.cvtColor(render_photo_np, cv2.COLOR_RGB2BGR)

    # ✅ Keep original sizes of input images
    image_with_red_h, image_with_red_w = image_with_red.shape[:2]
    overlayed_image_h, overlayed_image_w = overlayed_image.shape[:2]

    # Resize overlayed image only if necessary (to match height of image_with_red)
    if overlayed_image_h != image_with_red_h:
        scale_factor = image_with_red_h / overlayed_image_h
        overlayed_image_resized = cv2.resize(overlayed_image, 
                                             (int(overlayed_image_w * scale_factor), image_with_red_h))
    else:
        overlayed_image_resized = overlayed_image

    # Ensure both images are the same width (scale proportionally)
    if overlayed_image_resized.shape[1] != image_with_red_w:
        scale_factor = image_with_red_w / overlayed_image_resized.shape[1]
        overlayed_image_resized = cv2.resize(overlayed_image_resized, 
                                             (image_with_red_w, int(overlayed_image_resized.shape[0] * scale_factor)))

    # Create the final combined view (NO forced resizing)
    final_combined = cv2.hconcat([image_with_red, overlayed_image_resized])

    # Show exactly how it will appear in the video
    cv2.imshow("Unified Visualization", final_combined)

    return final_combined





def load_odomoetry_rosbag_info(odometry_bag_path,odometry_topic):
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

    return odometry_data

def load_odometry_from_txt(odometry_txt_path):
    odometry_data = []
    with open(odometry_txt_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 8:  # Ensure correct format
                timestamp = float(parts[0])
                position = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                orientation = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
                odometry_data.append((timestamp, position, orientation))

    # Sort odometry data by timestamp
    odometry_data.sort(key=lambda x: x[0])
    return odometry_data

# Visualize the saved maps **with consistent viewing settings**
def configure_view(vis, pcd):
    """Apply consistent Open3D visualization settings."""
    render_options = vis.get_render_option()
    render_options.point_size = 2.0  # Same fine visualization
    render_options.background_color = np.array([255, 255, 255])  # White background
    vis.add_geometry(pcd)

    # Set consistent camera settings
    view_control = vis.get_view_control()
    bounding_box = pcd.get_axis_aligned_bounding_box()
    center = bounding_box.get_center()
    view_control.set_lookat(center)
    view_control.set_front([0, -1, 0])  # Keep direction same
    view_control.set_up([0, 0, 1])  # Maintain Z-up
    view_control.set_zoom(1.5)  # Adjust for visibility

    vis.run()
    vis.destroy_window()

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
    IROS2024_Westhafen2LoopsSmallIsland

    '''

    lidar_bags_base_name_list = ['7_anlegen_80m_100kmph_BTUwDLR', '9_anlegen_80m_100kmph_BTUwDLR', '17_straight_200m_100kmph_BTUwDLR', '29_bridgecurve_80m_100kmph_BTUwDLR','30_bridgecurve_80m_100kmph_BTUwDLR','37_curvepromenade_160m_100kmph_BTUwDLR','53_schleuseeinfahrt_20m_100kmph_BTUwDLR','2023.11.07_friedrichstrasseToBerlinDom']






    if 1:
        lidar_bag_base_name = "30_bridgecurve_80m_100kmph_BTUwDLR"

    # for lidar_bag_base_name in lidar_bags_base_name_list:

        #Paths
        lidar_bag_path = "/home/knadmin/Ashwin/AURORA_dataset/Segmented3DMap/DATA/{}/{}.bag".format(lidar_bag_base_name,lidar_bag_base_name)
        odometry_bag_path = "/home/knadmin/Ashwin/AURORA_dataset/Segmented3DMap/DATA/{}/{}_trajectory.bag".format(lidar_bag_base_name,lidar_bag_base_name)
        odometry_txt_path = "/home/knadmin/Ashwin/AURORA_dataset/Segmented3DMap/DATA/{}/{}_trajectory.txt".format(lidar_bag_base_name, lidar_bag_base_name)
        original_bag_images_dump_folder = "/home/knadmin/Ashwin/AURORA_dataset/Segmented3DMap/DATA/{}/{}_images/images".format(lidar_bag_base_name,lidar_bag_base_name)
        lidaroverlay_images_dump_folder = "/home/knadmin/Ashwin/AURORA_dataset/Segmented3DMap/DATA/{}/{}_images/lidaroverlay_images".format(lidar_bag_base_name,lidar_bag_base_name)
        semantic_images_dump_folder = "/home/knadmin/Ashwin/AURORA_dataset/Segmented3DMap/DATA/{}/{}_images/semantic_images".format(lidar_bag_base_name,lidar_bag_base_name)
        output_txt_dir = "/home/knadmin/Ashwin/AURORA_dataset/Segmented3DMap/DATA/{}".format(lidar_bag_base_name)

        output_folder = os.path.join(output_txt_dir,"output_pcd")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_pcd_semantic_file = "{}/{}_trajectory_output_map_semantic.ply".format(output_folder,lidar_bag_base_name)
        output_pcd_photo_file = "{}/{}_trajectory_output_map_photo.ply".format(output_folder,lidar_bag_base_name)
        output_trajectory_pcd_file = "{}/{}_trajectory_output_map_trajectory.ply".format(output_folder,lidar_bag_base_name)
        filtered_pcd_path = "{}/{}_filtered_pcd.ply".format(output_folder,lidar_bag_base_name)
        class_labels_dump_filename = "{}/{}_class_labels.npy".format(output_folder,lidar_bag_base_name)
        # save_combined_window_video_filename_avi = "{}/{}_combined_view_points_avi.avi".format(output_folder,lidar_bag_base_name)
        # save_combined_window_video_filename_mp4 = "{}/{}_combined_view_points_mp4.mp4".format(output_folder,lidar_bag_base_name)
        # save_combined_window_video_filename_mjpg = "{}/{}_combined_view_points_MJPG.avi".format(output_folder,lidar_bag_base_name)
        # save_combined_window_video_filename_h264 = "{}/{}_combined_view_points_h264.mp4".format(output_folder,lidar_bag_base_name)
        # save_combined_window_video_filename_ffv1 = "{}/{}_combined_view_points_ffv1.avi".format(output_folder,lidar_bag_base_name)
        

        frame_width = 4416  # Adjust according to final window size
        frame_height = 1242
        # video_frame_width = 3840  # Adjust according to final window size
        # video_frame_height = 2160
        fps = 10 # Frames per second
        frame_queue = Queue(maxsize=10)  # Store up to 10 frames before processing


        # Define output file paths
        video_output_folder = output_folder
        os.makedirs(video_output_folder, exist_ok=True)

        # save_avi_xvid = os.path.join(video_output_folder, f"{lidar_bag_base_name}_XVID.avi")
        # save_avi_ffv1 = os.path.join(video_output_folder, f"{lidar_bag_base_name}_FFV1.avi")
        # save_avi_mjpg = os.path.join(video_output_folder, f"{lidar_bag_base_name}_MJPG.avi")
        # save_mp4 = os.path.join(video_output_folder, f"{lidar_bag_base_name}.mp4")

        save_combined_window_video_filename_avi = "{}/{}_combined_view_points_avi.avi".format(video_output_folder,lidar_bag_base_name)
        save_combined_window_video_filename_mp4 = "{}/{}_combined_view_points_mp4.mp4".format(video_output_folder,lidar_bag_base_name)
        save_combined_window_video_filename_mjpg = "{}/{}_combined_view_points_MJPG.avi".format(video_output_folder,lidar_bag_base_name)
        save_combined_window_video_filename_h264 = "{}/{}_combined_view_points_h264.mp4".format(video_output_folder,lidar_bag_base_name)
        save_combined_window_video_filename_ffv1 = "{}/{}_combined_view_points_ffv1.avi".format(video_output_folder,lidar_bag_base_name)

        # Define codecs
        fourcc_avi = cv2.VideoWriter_fourcc(*'XVID')  # Use 'XVID' for .avi or 'mp4v' for .mp4
        video_writer = cv2.VideoWriter(save_combined_window_video_filename_avi, fourcc_avi, fps, (frame_width, frame_height))
        fourcc_mp4v = cv2.VideoWriter_fourcc(*'mp4v') 
        video_writer_mp4 = cv2.VideoWriter(save_combined_window_video_filename_mp4, fourcc_mp4v, fps, (frame_width, frame_height))
        fourcc_mjpg = cv2.VideoWriter_fourcc(*'MJPG')  
        video_writer_mjpg = cv2.VideoWriter(save_combined_window_video_filename_mjpg, fourcc_mjpg, fps, (frame_width, frame_height))
        fourcc_h264 = cv2.VideoWriter_fourcc(*'H264')
        video_writer_h264 = cv2.VideoWriter(save_combined_window_video_filename_h264, fourcc_h264, fps, (frame_width, frame_height))
        fourcc_ffv1 = cv2.VideoWriter_fourcc(*'FFV1')
        video_writer_ffv1 = cv2.VideoWriter(save_combined_window_video_filename_ffv1, fourcc_ffv1, fps, (frame_width, frame_height))





        #TOpic names
        lidar_topic = "/VLP32/velodyne_points"
        # lidar_topic = "/ouster/points"
        left_image_topic = "/zed2i/zed_node/left/image_rect_color/compressed"
        right_image_topic = "/zed2i/zed_node/right/image_rect_color/compressed"

        odometry_topic = "/lio_sam/mapping/odometry"
        

        trained_models_Save_path = "/home/knadmin/Ashwin/Semantic_labelled_by_Lukas_Hosch/trained_models"
        # trained_models_Save_path = "/home/knadmin/Ashwin/AURORA_dataset/Segmented3DMap"
        # model_name = "segformer-best_6classes_aug_adjustable_lr_customweights"
        model_name = "segformer-itr2_6classes_aug_adjustable_lr_customweights"
        model_save_path = os.path.join(trained_models_Save_path,model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
        # Load the best saved model before inference
        model = SegformerForSemanticSegmentation.from_pretrained(model_save_path).to(device)
        model.eval()  # Ensure model is in evaluation mode
        model.to(device)
        # CATEGORY_COLORS = load_category_colors()
        CATEGORY_COLORS = {
        0: ([135, 206, 250], "Sky"),
        1: ([0, 191, 255], "Water"),
        2: ([50, 205, 50], "Vegetation"),
        3: ([200, 0, 0], "Riverbank"),
        4: ([184, 134, 11], "Bridge"),
        5: ([157, 0, 255], "Other")
        }
        camera_intrinsics = load_camera_intrinsics()
        selected_classes = [1, 2, 4]  #for filtering specific classes in the filtered pointcloud
        #{"0": {"color": [135, 206, 250], "name": "Sky"}, "1": {"color": [0, 191, 255], "name": "Water"}, "2": {"color": [50, 205, 50], "name": "Vegetation"}, "3": {"color": [34, 139, 34], "name": "Riverbank"}, "4": {"color": [184, 134, 11], "name": "Bridge"}, "5": {"color": [157, 0, 255], "name": "Other"}}

        trajectory_flag = False
        apply_ransac_flag = False
        compute_map_entropy_flag = False
        downsample_pcd_flag = False
        save_images_flag = False
        generate_pcd_flag = True
        resize_flag = False
        load_data_mode = "rosbag_mode"  #"txt_file_mode"  #"rosbag_mode"

        frame_count = 0  # Counter for frame-based naming

        # if os.path.exists(output_txt_dir):
        #     shutil.rmtree(output_txt_dir)  
        os.makedirs(output_txt_dir, exist_ok=True)  # Create new empty folder
        
        # cv2.namedWindow("Original left Camera Image",cv2.WINDOW_NORMAL)
        cv2.namedWindow("Original left Camera Image",cv2.WINDOW_NORMAL)
        cv2.namedWindow("Original Image with lidar points in red projected", cv2.WINDOW_NORMAL) 
        cv2.namedWindow("Only Lidar points with Photo colour", cv2.WINDOW_NORMAL) 
        cv2.namedWindow("Semantic Image", cv2.WINDOW_NORMAL) 
        cv2.namedWindow("Alpha Blended Image",cv2.WINDOW_NORMAL)

        cv2.namedWindow("Unified Visualization", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Unified Visualization", frame_width, frame_height)  # Set initial size

        # Visualization Configuration
        vis_config = {
            "window_width": 1600,
            "window_height": 900,
            "point_size": 1.0,  # Fine visualization
            "background_color": [255, 255, 255],  # Black background
            "show_coordinate_frame": False,
            "point_color_option": "Color",  # Color based on Z-coordinate (height/depth)
            "add_horizontal_plane": False,
            "rectangle": {
                "bottom_left": [0.0, 0.0, -3.0],  # Height for horizontal plane
                "top_right": [0.0, 0.0, -3.0],
                "color": [0.2, 0.2, 0.2]
            }
        }

        # cv2.namedWindow("Depth Image",cv2.WINDOW_NORMAL)

        # Velodyne LiDAR to Camera Transformation Matrix
        R = np.array([[-0.0086, 0.0068, 0.9999],
                    [-1.0000, -0.0006, -0.0086],
                    [0.0006, -1.0000, 0.0068]])

        # translation = np.array([0.0441, 0.0649, -0.0807]) #From matlab lidar toolbox calibration
        # translation = np.array([0.059, 0.06, 0.095]) #Iulian's estimates
        translation = np.array([0.0441, 0.0649, -0.0807]) #my_estimates

        # Ouster LiDAR to Camera Transformation Matrix
        # R = np.array([[-0.0308, 0.0251, 0.9992],
        #             [-0.9995, -0.0108, -0.0305],
        #             [0.0101, -0.9996, 0.0254]])
        # R = np.array([[-0.0086, 0.0068, 0.9999],
        #     [-1.0000, -0.0006, -0.0086],
        #     [0.0006, -1.0000, 0.0068]])
        # translation = np.array([-3.5, -0.3797, -1.9244])  # My estimates



        def rigid_transform(points, R, translation):
            """Apply rigid transformation to the point cloud."""
            return np.dot(points, R) + translation

        bridge = CvBridge()

        start_time = time.time()

        # Initialize empty Open3D point cloud
        pcd_live_semantic = o3d.geometry.PointCloud()
        pcd_live_photo = o3d.geometry.PointCloud()
        # Setup Open3D Visualizer

        # Setup Open3D Visualizer
        vis_semantic = o3d.visualization.Visualizer()
        vis_semantic.create_window(window_name="Live Semantic Point Cloud",
                                width=vis_config["window_width"],
                                height=vis_config["window_height"])

        render_options = vis_semantic.get_render_option()
        render_options.point_size = vis_config["point_size"]
        render_options.background_color = np.array(vis_config["background_color"])
        render_options.show_coordinate_frame = vis_config["show_coordinate_frame"]

        # Set point color option based on configuration
        render_options.point_color_option = o3d.visualization.PointColorOption.Color


        
        vis_semantic.add_geometry(pcd_live_semantic)
        # apply_saved_view(vis_semantic)
        # view_control = vis_semantic.get_view_control()
        # # Define a fixed view (example parameters)
        # view_control.set_lookat([0, 0, 0])  # Center of the scene
        # view_control.set_up([0, -1, 0])     # Direction of the 'up' vector
        # view_control.set_front([1, 0, 0])   # Direction the camera faces
        # view_control.set_zoom(0.8)          # Zoom level (1.0 means default distance)


        # Setup Open3D Visualizer for Photorealistic PCD
        vis_photo = o3d.visualization.Visualizer()
        vis_photo.create_window(window_name="Live Photo Point Cloud",
                                width=vis_config["window_width"],
                                height=vis_config["window_height"],
                                visible=True)

        render_options_photo = vis_photo.get_render_option()
        render_options_photo.point_size = vis_config["point_size"]
        render_options_photo.background_color = np.array(vis_config["background_color"])
        render_options_photo.show_coordinate_frame = vis_config["show_coordinate_frame"]

        # Set point color option to match semantic PCD visualization
        render_options_photo.point_color_option = o3d.visualization.PointColorOption.Color
        
        vis_photo.add_geometry(pcd_live_photo)
        # apply_saved_view(vis_photo)
        # Apply consistent camera view
        # view_control_photo = vis_photo.get_view_control()
        # view_control_photo.set_lookat([0, 0, 0])
        # view_control_photo.set_up([0, -1, 0])
        # view_control_photo.set_front([0.5, -0.8, 0.5])
        # view_control_photo.set_zoom(3)



        # Load odometry from ROS bag with high-precision timestamps
        # odometry_data = []
        # odometry_bag = rosbag.Bag(odometry_bag_path, "r")
        # for topic, msg, t in odometry_bag.read_messages(topics=[odometry_topic]):
        #     timestamp = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9  # High-precision timestamp
        #     position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        #     orientation = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        #     odometry_data.append((timestamp, position, orientation))
        # odometry_bag.close()
        if load_data_mode == 'rosbag_mode':
            odometry_data = load_odomoetry_rosbag_info(odometry_bag_path,odometry_topic)
        elif load_data_mode == 'txt_file_mode':
            odometry_data = load_odometry_from_txt(odometry_txt_path)




        odometry_timestamps = [x[0] for x in odometry_data]

        # Process LiDAR and Image data together
        pcd_points = []
        photo_colors_list = []
        semantic_colors_list = []
        intensity_list = []
        ring_list = []
        class_labels_list = []
        time_stamp_list = []

        lidar_bag = rosbag.Bag(lidar_bag_path, "r")
        left_image = None  # Initialize image storage
        right_image = None  # Initialize image storage
        # depth_image = None

        points_counter =0
        image_counter = 0

        # During the visualization update loop
        vis_semantic.update_geometry(pcd_live_semantic)
        vis_semantic.poll_events()
        vis_semantic.update_renderer()

        # Add horizontal plane if configured
        if vis_config["add_horizontal_plane"]:
            min_bound = pcd_live_semantic.get_min_bound()
            max_bound = pcd_live_semantic.get_max_bound()

            bottom_left = [
                min_bound[0],
                min_bound[1],
                vis_config["rectangle"]["bottom_left"][2],
            ]
            top_right = [
                max_bound[0],
                max_bound[1],
                vis_config["rectangle"]["top_right"][2],
            ]
            rectangle = create_rectangle(bottom_left, top_right, vis_config["rectangle"]["color"])
            vis_semantic.add_geometry(rectangle)



        frame_skip_rate = 1  # Process every 5th frame
        frame_index = 0  # Counter for frames

        os.makedirs(lidaroverlay_images_dump_folder, exist_ok=True)
        os.makedirs(semantic_images_dump_folder, exist_ok=True)

        # Tracking variables
        latest_left_image = None
        latest_image_timestamp = None
        num_lidar_frames_processed = 0  # Count LiDAR frames for the current image
        skip_next_image = False  # Flag to skip every other image

        topic_info = lidar_bag.get_type_and_topic_info()
        total_lidar_messages = topic_info.topics[lidar_topic].message_count
        total_lidar_messages_to_be_processed = (total_lidar_messages/4)

        for topic, msg, t in lidar_bag.read_messages(topics=[lidar_topic, left_image_topic, right_image_topic]):
            timestamp = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9  # High-precision timestamp



            
            if topic == left_image_topic:



                if skip_next_image:
                    skip_next_image = False  # Reset flag
                    continue  # Skip this image

                # if frame_index%2 == 0:
                #     continue
                print("FRAME COUNT",frame_index)
                latest_image_timestamp = timestamp
                num_lidar_frames_processed = 0  # Reset LiDAR frame count
                skip_next_image = True  # Set flag to skip the next camera frame
                
                left_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                if resize_flag:
                    left_image = cv2.resize(left_image, (left_image.shape[1] * 2, left_image.shape[0] * 2), interpolation=cv2.INTER_LINEAR)

                image_counter+=1
                cv2.imshow("Original left Camera Image", left_image)
                if save_images_flag:
                    os.makedirs(original_bag_images_dump_folder, exist_ok=True)
                    img_path = os.path.join(original_bag_images_dump_folder,"image_{}.png".format(timestamp))
                    print("img_path:",img_path)
                    cv2.imwrite(img_path,left_image)
                continue  # Store the latest image and move on


            if topic == lidar_topic and left_image is not None:
                # Only process 2 LiDAR frames per image
                if num_lidar_frames_processed >= 1:
                    continue  # Skip extra LiDAR frames

                num_lidar_frames_processed += 1  # Count LiDAR frames used for this image

                frame_inference_start_time = time.time()
                # Find the closest odometry timestamp
                closest_idx = np.argmin(np.abs(np.array(odometry_timestamps) - timestamp))
                closest_time, position, orientation = odometry_data[closest_idx]





                
                print("**************************")
                print("Frame counter: {}/{}".format(frame_count,int(total_lidar_messages_to_be_processed)))
                
                print(f"Processing LiDAR at {timestamp} -> Closest odometry at {closest_time} (Δt = {abs(closest_time - timestamp):.3f}s)")
                
                # Convert LiDAR message to numpy array- every single lidar point in this topic
                lidar_points = np.array(list(pc2.read_points(
                        msg, field_names=("x", "y", "z", "intensity", "ring", "time"), skip_nans=True)))
                print("Available LiDAR fields:", [f.name for f in msg.fields])

                # import pdb;pdb.set_trace()
                
                # Extract x, y, z, intensity, and ring separately
                points = lidar_points[:, :3]
                intensity = lidar_points[:, 3]
                ring = lidar_points[:, 4]
                time_stamp_within_roation = lidar_points[:, 5]


                
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
                # print("Per Frame inference Time (without PCD part)",frame_inference_duration)

                #Part of the code to use for using semantic iamge color
                colored_points_semantic, class_labels, valid_indices = colorize_point_cloud_semantic(lidar_points, left_image, transformed_points, semantic_predictions)
                # class_labels_array = np.array(class_labels).reshape(-1, 1)
                

                # Overlay segmentation mask on original image
                alpha = 0.5
                overlayed_image = cv2.addWeighted(left_image, 1, semantic_color_mask, alpha, 0)


                # Save transformed + colorized LiDAR points to TXT
                save_pointcloud_to_txt(frame_count, transformed_points, colored_points_photo, valid_indices, timestamp, lidar_points, output_folder, mode='photo')
                save_pointcloud_to_txt(frame_count, transformed_points, colored_points_semantic, valid_indices, timestamp, lidar_points, output_folder, mode='semantic')

                

                

                img_path_lidaroverlay = os.path.join(lidaroverlay_images_dump_folder,"image_{}.png".format(frame_count))
                img_path_semantic = os.path.join(semantic_images_dump_folder,"image_{}.png".format(frame_count))
                
                cv2.imwrite(img_path_lidaroverlay,image_with_red)
                cv2.imwrite(img_path_semantic,overlayed_image)
                # import pdb;pdb.set_trace()
                frame_count += 1  # Increment frame count


                
                valid_points = points[valid_indices] 
            

                # colors = np.zeros((points.shape[0], 3), dtype=np.uint8)

                print("checking length", len(colored_points_semantic), len(valid_indices))
                points_counter += len(valid_indices)
                # import pdb;pdb.set_trace()

                black_image = overlay_points_on_black_image(x_proj, y_proj, left_image)

                cv2.imshow("Original Image with lidar points in red projected",image_with_red) 
                cv2.imshow("Only Lidar points with Photo colour", black_image) 
                cv2.imshow("Semantic Image", semantic_color_mask) 
                cv2.imshow("Alpha Blended Image", overlayed_image) 
                cv2.waitKey(1) 
                
                # continue

                pcd_build_start_time = time.time()
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
                current_points = np.zeros((len(valid_indices), 3))  # Initialize color array


                total_points = len(valid_indices)  # Total number of valid colorized points
                non_black_count = 0  # Counter for non-black points

                new_points = []
                new_colors = []

                for idx, point in enumerate(colored_points_semantic):
                    x, y, z, r, g, b = point  # Unpack each point with color values

                    # Normalize RGB values (Open3D expects colors in range [0,1])
                    r_norm = r / 255.0
                    g_norm = g / 255.0
                    b_norm = b / 255.0
                    current_points[idx] = [x, y, z]
                    semantic_colors[idx] = [r_norm, g_norm, b_norm]
                    # new_points.append([x, y, z])
                    # new_colors.append([r / 255.0, g / 255.0, b / 255.0])

                    # pcd_live.points.extend(np.random.rand(n_new, 3))
                    # colors[idx] = [1, 0, 0]

                for idx, point in enumerate(colored_points_photo):
                    x, y, z, r, g, b = point  # Unpack each point with color values

                    # Normalize RGB values (Open3D expects colors in range [0,1])
                    r_norm = r / 255.0
                    g_norm = g / 255.0
                    b_norm = b / 255.0

                    photo_colors[idx] = [r_norm, g_norm, b_norm]
                    # colors[idx] = [1, 0, 0]


                # Convert to Open3D format
                new_points = np.array([pt[:3] for pt in colored_points_semantic])
                new_colors = np.array([pt[3:] for pt in colored_points_semantic]) 
                



            
                # Append to global lists
                pcd_points.append(transformed_points)
                semantic_colors = semantic_colors.astype(np.float32)
                photo_colors = photo_colors.astype(np.float32)
                semantic_colors_list.append(semantic_colors)
                photo_colors_list.append(photo_colors)
                intensity_list.append(intensity)
                ring_list.append(ring)
                time_stamp_list.append(time_stamp_within_roation)
                class_labels_list.extend(class_labels) 



                # if len(new_points) > 0:
                pcd_live_semantic.points = o3d.utility.Vector3dVector(np.vstack(pcd_points))           
                pcd_live_semantic.colors = o3d.utility.Vector3dVector(np.vstack(semantic_colors_list))

                pcd_live_photo.points = o3d.utility.Vector3dVector(np.vstack(pcd_points))           
                pcd_live_photo.colors = o3d.utility.Vector3dVector(np.vstack(photo_colors_list))

                
                combined_video_Stream_resized = combine_visualization(vis_semantic, image_with_red, vis_photo, overlayed_image)
                # combined_video_Stream_resized = cv2.resize(combined_video_Stream, (frame_width, frame_height))
                print("combined_video_Stream_resized",combined_video_Stream_resized.shape)

                # Write to video files
                if video_writer.isOpened():
                    video_writer.write(combined_video_Stream_resized)
                else:
                    print("Error: AVI Video writer not opened!")

                if video_writer_mp4.isOpened():
                    video_writer_mp4.write(combined_video_Stream_resized)
                else:
                    print("Error: MP4 Video writer not opened!")

                if video_writer_mjpg.isOpened():
                    video_writer_mjpg.write(combined_video_Stream_resized)
                else:
                    print("Error: MJPG Video writer not opened!")

                if video_writer_h264.isOpened():
                    video_writer_h264.write(combined_video_Stream_resized)
                else:
                    print("Error: H264 Video writer not opened!")

                if video_writer_ffv1.isOpened():
                    video_writer_ffv1.write(combined_video_Stream_resized)
                else:
                    print("Error: FFV1 Video writer not opened!")



                # import pdb;pdb.set_trace()
                vis_semantic.update_geometry(pcd_live_semantic)
                vis_semantic.poll_events()
                vis_semantic.update_renderer()
                vis_semantic.reset_view_point(True) 

                vis_photo.update_geometry(pcd_live_photo)
                vis_photo.poll_events()
                vis_photo.update_renderer()
                vis_photo.reset_view_point(True) 

                pcd_build_end_time = time.time()
                pcd_build_duration = pcd_build_end_time - pcd_build_start_time
                frame_and_pcd_corresponding_process_time = pcd_build_end_time - frame_inference_start_time
                print("Per Frame inference Time (without PCD part)",frame_inference_duration)
                print("PCD Build Time per Frame",pcd_build_duration)        
                print("PFrame + PCD process per corresponding parts",frame_and_pcd_corresponding_process_time)


                


        video_writer.release()
        video_writer_h264.release()
        video_writer_mjpg.release()
        video_writer_mp4.release()
        video_writer_ffv1.release()
        cv2.destroyAllWindows()
        # vis.destroy_window()
        print("************************")

        # Convert to Open3D format
        pcd_points = np.vstack(pcd_points)
        semantic_colors_list = np.vstack(semantic_colors_list)
        photo_colors_list = np.vstack(photo_colors_list)
        intensity_list = np.hstack(intensity_list)[:, None]  # Ensure correct shape
        ring_list = np.hstack(ring_list)[:, None]
        time_stamp_list = np.hstack(time_stamp_list)[:, None]
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

        np.save(class_labels_dump_filename, class_labels_array)
        end_time = time.time()
        time_taken = end_time - start_time
        print("Total TIme taken",time_taken)

        if generate_pcd_flag:
            # Save the final point clouds
            print(f"Saving final Semantic 3D map to {output_pcd_semantic_file}")
            
            if trajectory_flag:
                o3d.io.write_point_cloud(output_pcd_semantic_file, combined_pcd_semantic)
                o3d.io.write_point_cloud(output_pcd_photo_file, combined_pcd_photo)
                o3d.io.write_point_cloud(output_trajectory_pcd_file, trajectory_pcd)



                # Show each PCD with the fixed viewing configuration
                vis1 = o3d.visualization.Visualizer()
                vis1.create_window(window_name="Semantic 3D Map + Boat Trajectory")
                configure_view(vis1, combined_pcd_semantic)

                vis2 = o3d.visualization.Visualizer()
                vis2.create_window(window_name="Photorealistic Map + Boat Trajectory")
                configure_view(vis2, combined_pcd_photo)

                vis3 = o3d.visualization.Visualizer()
                vis3.create_window(window_name="Boat Trajectory")
                configure_view(vis3, trajectory_pcd)

                # ✅ Filter and visualize selected classes
                filtered_pcd = filter_pcd_by_class(pcd_semantic, class_labels_dump_filename, selected_classes, CATEGORY_COLORS)
                o3d.io.write_point_cloud(filtered_pcd_path, filtered_pcd)
                print("✅ Filtered PCD saved as filtered_pcd.ply")

                vis4 = o3d.visualization.Visualizer()
                vis4.create_window(window_name="Filtered Pointcloud")
                configure_view(vis4, filtered_pcd)

            else:
                o3d.io.write_point_cloud(output_pcd_semantic_file, pcd_semantic)
                o3d.io.write_point_cloud(output_pcd_photo_file, pcd_photo)

                # Load and apply the visualization config
                config_path = "config/pcd_config.yaml"
                config = load_config(config_path)
                
                # Visualize the PCDs using the same settings as during live processing
                map_cloud_semantic = o3d.io.read_point_cloud(output_pcd_semantic_file)   
                visualize_mapcloud(map_cloud_semantic, config)

                map_cloud_photo = o3d.io.read_point_cloud(output_pcd_photo_file)   
                visualize_mapcloud(map_cloud_photo, config)

                # ✅ Filter and visualize selected classes
                filtered_pcd = filter_pcd_by_class(pcd_semantic, class_labels_dump_filename, selected_classes, CATEGORY_COLORS)
                o3d.io.write_point_cloud(filtered_pcd_path, filtered_pcd)
                print("✅ Filtered PCD saved as filtered_pcd.ply")

                vis5 = o3d.visualization.Visualizer()
                vis5.create_window(window_name="Filtered Pointcloud")
                configure_view(vis5, filtered_pcd)

        # Close lidar bag after processing
        lidar_bag.close()
        print("Visualization running... Press Q to exit.")








    #2432116



        



