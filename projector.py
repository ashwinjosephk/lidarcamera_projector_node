#!/usr/bin/env python3

import rospy
import message_filters
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, CompressedImage
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointField
import sensor_msgs.point_cloud2 as pc2
from threading import Lock
import scipy.spatial
# import threading
import os
import subprocess
import re

# Projection parameters
fx, fy = 1089.8, 1086
cx, cy = 1150.3, 638.8

# Project pointcloud onto img
# Project pointcloud onto img
R = np.array([[-0.0086, 0.0068, 0.9999],
              [-1.0000, -0.0006, -0.0086],
              [0.0006, -1.0000, 0.0068]])

# translation = np.array([0.0441, 0.0649, -0.0807]) #From matlab lidar toolbox calibration
# translation = np.array([0.059, 0.06, 0.095]) #Iulian's estimates
translation = np.array([0.0441, 0.0649, -0.0807]) #my_estimates

bridge = CvBridge()

def rigid_transform(points, R, translation):
    """Apply rigid transformation to the point cloud."""
    rospy.loginfo("Applying rigid transformation to LiDAR points.")
    return np.dot(points, R) + translation

def colorize_point_cloud(lidar_points, image, transformed_points):
    """Colorize the point cloud based on the image colors with correct RGB values."""
    rospy.loginfo("Colorizing the point cloud with RGB values.")

    Xc, Yc, Zc = transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2]
    
    # Step 1: Find valid points that are in front of the camera (Z > 0)
    valid_indices = np.where(Zc > 0)[0]  # Get indices directly to maintain order
    Xc, Yc, Zc = Xc[valid_indices], Yc[valid_indices], Zc[valid_indices]

    # Step 2: Project the valid points into the image plane
    x_proj = (fx * Xc / Zc + cx).astype(int)
    y_proj = (fy * Yc / Zc + cy).astype(int)

    # Step 3: Find points that fall within the image bounds
    inside_image = (x_proj >= 0) & (x_proj < image.shape[1]) & (y_proj >= 0) & (y_proj < image.shape[0])
    
    # Step 4: Apply the mask to retain only valid points (ensuring order is kept)
    valid_indices = valid_indices[inside_image]  # Filter indices based on image bounds
    x_proj, y_proj = x_proj[inside_image], y_proj[inside_image]

    # Step 5: Extract RGB colors
    colors = image[y_proj, x_proj]

    # Step 6: Create colorized points list while **maintaining original order**
    colored_points = [
        (lidar_points[i][0], lidar_points[i][1], lidar_points[i][2], 
         (int(colors[j][0]) << 16) | (int(colors[j][1]) << 8) | int(colors[j][2]))  # Pack RGB
        for j, i in enumerate(valid_indices)
    ]

    rospy.loginfo(f"Successfully colorized {len(colored_points)} points.")
    return colored_points, valid_indices  # Return valid_indices for correct `.txt` saving




def get_running_rosbag_name():
    """
    Automatically detects the running rosbag file name by checking active ROS nodes.
    Returns the detected rosbag name or "rosbag_unknown" if not found.
    """
    try:
        output = subprocess.check_output(["rosnode", "info", "/rosbag_play"], stderr=subprocess.DEVNULL, text=True)
        match = re.search(r"rosbag play ([\w\-\.]+)", output)
        if match:
            rosbag_name = match.group(1).replace(".bag", "")  # Remove .bag extension
            rospy.loginfo(f"Detected rosbag name: {rosbag_name}")
            return rosbag_name
    except subprocess.CalledProcessError:
        rospy.logwarn("Could not detect rosbag name automatically. Using default.")
    return "rosbag_unknown"




def overlay_points_on_black_image(u, v, image, projected_image_publisher):
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
    projected_img_msg = bridge.cv2_to_imgmsg(black_image, encoding="bgr8")
    projected_image_publisher.publish(projected_img_msg)

    rospy.loginfo("Published projected LiDAR points onto a black image.")











class LidarCameraProjection:
    def __init__(self):
        rospy.init_node("lidar_camera_projection_node")

        self.lidar_topic = rospy.get_param("~lidar_topic", "/VLP32/velodyne_points")
        self.camera_topic_base = rospy.get_param("~camera_topic_base", "/zed2i/zed_node/left/image_rect_color")
        self.use_compressed = rospy.get_param("~use_compressed", True)
        self.use_img_resize = rospy.get_param("~use_img_resize", False)
        self.output_topic = rospy.get_param("~output_topic", "/projected_points_image")
        self.mask_topic = rospy.get_param("~mask_topic", "/projected_points_mask")
        self.colorized_topic = rospy.get_param("~colorized_topic", "/colorized_pointcloud")
        self.projected_image_topic = rospy.get_param("~projected_lidar_image", "/projected_lidar_image")

        self.rosbag_name = get_running_rosbag_name()
        self.txt_output_dir = "berlinDom"


        self.time_tolerance = rospy.get_param("~time_tolerance", 0.03)
        self.processing_rate = rospy.get_param("~processing_rate", 10)  # Hz
        # self.skip_frames = rospy.get_param("~skip_frames", 5)

        rospy.loginfo("Initializing publishers and subscribers.")

        self.colorized_publisher = rospy.Publisher(self.colorized_topic, PointCloud2, queue_size=1)
        self.projection_publisher = rospy.Publisher(self.output_topic, Image, queue_size=1)
        self.mask_publisher = rospy.Publisher(self.mask_topic, Image, queue_size=1)
        self.projected_image_publisher = rospy.Publisher(self.projected_image_topic, Image, queue_size=1)

        if self.use_compressed:
            self.camera_topic = self.camera_topic_base + "/compressed"
            self.image_sub = message_filters.Subscriber(self.camera_topic, CompressedImage)
        else:
            self.camera_topic = self.camera_topic_base
            self.image_sub = message_filters.Subscriber(self.camera_topic, Image)

        self.lidar_sub = message_filters.Subscriber(self.lidar_topic, PointCloud2)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub],
            queue_size=3,
            slop=self.time_tolerance
        )
        self.ts.registerCallback(self.callback)

        self.lock = Lock()
        self.latest_data = None
        self.frame_count = 0

        rospy.Timer(rospy.Duration(1.0 / self.processing_rate), self.process_data)

        rospy.loginfo("LidarCameraProjection node initialized.")


    def callback(self, image_msg, lidar_msg):
        """
        Callback function for synchronized LiDAR and camera image messages.
        Stores the latest data in a thread-safe manner.
        """
        with self.lock:
            self.latest_data = (image_msg, lidar_msg)
        rospy.loginfo("Received synchronized LiDAR and image messages.")


    def project_points(self,lidar_points, image, projection_publisher, mask_publisher, colorized_publisher, projected_image_publisher, image_msg):
        try:
            rospy.loginfo("Starting point projection.")
            transformed_points = rigid_transform(lidar_points[:, :3], R, translation)

            Xc, Yc, Zc = transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2]
            rospy.loginfo(f"Number of valid points before filtering: {len(Xc)}")

            valid_indices = Zc > 0
            Xc, Yc, Zc = Xc[valid_indices], Yc[valid_indices], Zc[valid_indices]
            rospy.loginfo(f"Number of valid points after filtering: {len(Xc)}")


            #2D image projection formula got from matlab lidar camera projection tutorial website
            x_proj = (fx * Xc / Zc + cx).astype(int)
            y_proj = (fy * Yc / Zc + cy).astype(int)


            #checking if projected 2d points are within image boundary
            valid_points = (x_proj >= 0) & (x_proj < image.shape[1]) & (y_proj >= 0) & (y_proj < image.shape[0])
            x_proj, y_proj = x_proj[valid_points], y_proj[valid_points]

            rospy.loginfo(f"Number of valid projected points on the image: {len(x_proj)}")


            #*************
            # 1️⃣ Create the original image with red-projected points
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            image_with_red = image.copy()
            point_size = 2
            for x, y in zip(x_proj, y_proj):
                x_start, x_end = max(0, x - point_size), min(image.shape[1], x + point_size + 1)
                y_start, y_end = max(0, y - point_size), min(image.shape[0], y + point_size + 1)
                image_with_red[y_start:y_end, x_start:x_end] = [0, 0, 255]  # Red color
                mask[y_start:y_end, x_start:x_end] = 255  # White mask

            # Publish the projected image and mask
            projected_image_msg = bridge.cv2_to_imgmsg(image_with_red, encoding="bgr8")
            projection_publisher.publish(projected_image_msg)

            mask_msg = bridge.cv2_to_imgmsg(mask, encoding="mono8")
            mask_publisher.publish(mask_msg)

            rospy.loginfo("Published projected image with red points and mask.")
             #*************



             #*************
            # 2️⃣ Publish the colorized point cloud (for RViz)
            colored_points, valid_indices  = colorize_point_cloud(lidar_points, image, transformed_points)
            fields = [
                PointField("x", 0, PointField.FLOAT32, 1),
                PointField("y", 4, PointField.FLOAT32, 1),
                PointField("z", 8, PointField.FLOAT32, 1),
                PointField("rgb", 12, PointField.UINT32, 1),
            ]
             #*************




             #*************
            # Save the point cloud to a file
            self.save_pointcloud_to_txt(lidar_points, colored_points, valid_indices, image_msg.header.stamp.to_sec())
             #*************

             #************* Publish colorized pointcloud
            header = rospy.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "velodyne_VLP32"

            colorized_pc2 = pc2.create_cloud(header, fields, colored_points)
            colorized_publisher.publish(colorized_pc2)
             #*************


             #*************
            # 3️⃣ Publish projected and colorsied points on a black image (RGB-colored points)
            overlay_points_on_black_image(x_proj, y_proj, image, projected_image_publisher)
             #*************





        except Exception as e:
            rospy.logerr(f"Error in project_points: {e}")

    

    def save_pointcloud_to_txt(self, lidar_points, colored_points, valid_indices, timestamp):
        """
        Saves the processed colorized point cloud to a text file.
        Ensures only front-facing (Z > 0) points are saved with correct order matching.
        """
        
        os.makedirs(self.txt_output_dir, exist_ok=True)
        self.frame_count += 1  
        filename = os.path.join(self.txt_output_dir, f"{self.txt_output_dir}_pcd_{self.frame_count:06d}.txt")

        # Filter LiDAR points based on valid_indices
        filtered_lidar_points = lidar_points[valid_indices]

        with open(filename, "w") as f:
            for i in range(len(filtered_lidar_points)):
                x, y, z, intensity, ring, time_within_rotation = filtered_lidar_points[i]
                _, _, _, rgb = colored_points[i]  # Extract RGB values
                r = (rgb >> 16) & 0xFF
                g = (rgb >> 8) & 0xFF
                b = rgb & 0xFF

                f.write(f"{x:.6f},{y:.6f},{z:.6f},{intensity:.2f},{int(ring)},{time_within_rotation:.6f},{r},{g},{b}\n")

        rospy.loginfo(f"Saved corrected colorized point cloud to {filename}")


    def process_data(self, event):
        # threading.Thread(target=self.process_data_thread, daemon=True).start()
        self.process_data_thread()


    def process_data_thread(self):
        """ Runs LiDAR processing in a separate thread to prevent blocking ROS. """
        with self.lock:
            if self.latest_data is None:
                rospy.loginfo("No synchronized data available for processing.")
                return
            image_msg, lidar_msg = self.latest_data
            self.latest_data = None

        try:
            rospy.loginfo("Processing synchronized data.")

            if self.use_compressed:
                np_arr = np.frombuffer(image_msg.data, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                image = bridge.imgmsg_to_cv2(image_msg, "bgr8")

            if self.use_img_resize:
                image = cv2.resize(image, (image.shape[1]*2, image.shape[0]*2))

            lidar_points = np.array(list(pc2.read_points(
                lidar_msg, field_names=("x", "y", "z", "intensity", "ring", "time"), skip_nans=True)))

            rospy.loginfo(f"LiDAR points count: {lidar_points.shape[0]}")
            rospy.loginfo(f"Name of rosbag automatically detected: {self.rosbag_name}")
            # 🏎️ Faster densification
            # lidar_points = densify_pointcloud(lidar_points, num_extra_points=2)

            # rospy.loginfo(f"Densified LiDAR points count: {lidar_points.shape[0]}")

            self.project_points(
                lidar_points, 
                image, 
                self.projection_publisher, 
                self.mask_publisher, 
                self.colorized_publisher, 
                self.projected_image_publisher,
                image_msg
            )



        except Exception as e:
            rospy.logerr(f"Error in process_data: {e}")



def main():
    try:
        rospy.loginfo("Starting LidarCameraProjection node.")
        LidarCameraProjection()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("LidarCameraProjection node shut down.")

if __name__ == "__main__":
    main()