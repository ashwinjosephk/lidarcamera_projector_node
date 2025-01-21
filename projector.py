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

# Projection parameters
fx, fy = 1089.8, 1086
cx, cy = 1150.3, 638.8

# Project pointcloud onto img
# Project pointcloud onto img
R = np.array([[-0.0086, 0.0068, 0.9999],
              [-1.0000, -0.0006, -0.0086],
              [0.0006, -1.0000, 0.0068]])
# translation = np.array([0.0441, 0.0649, -0.0807])
translation = np.array([0.059, 0.06, 0.095])

bridge = CvBridge()

def rigid_transform(points, R, translation):
    """Apply rigid transformation to the point cloud."""
    rospy.loginfo("Applying rigid transformation to LiDAR points.")
    return np.dot(points, R) + translation

def colorize_point_cloud(lidar_points, image, transformed_points):
    """Colorize the point cloud based on the image colors."""
    Xc, Yc, Zc = transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2]
    valid_indices = Zc > 0
    Xc, Yc, Zc = Xc[valid_indices], Yc[valid_indices], Zc[valid_indices]

    x_proj = (fx * Xc / Zc + cx).astype(int)
    y_proj = (fy * Yc / Zc + cy).astype(int)

    valid_points = (x_proj >= 0) & (x_proj < image.shape[1]) & (y_proj >= 0) & (y_proj < image.shape[0])
    x_proj, y_proj = x_proj[valid_points], y_proj[valid_points]

    colors = image[y_proj, x_proj]
    colored_points = []

    for i, point in enumerate(lidar_points[valid_indices][valid_points]):
        x, y, z = point[:3]
        r, g, b = colors[i]
        colored_points.append((x, y, z, r / 255.0, g / 255.0, b / 255.0))

    return colored_points

def project_points(lidar_points, image, projection_publisher, mask_publisher, colorized_publisher):
    try:
        rospy.loginfo("Starting point projection.")
        transformed_points = rigid_transform(lidar_points[:, :3], R, translation)

        Xc, Yc, Zc = transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2]
        valid_indices = Zc > 0
        Xc, Yc, Zc = Xc[valid_indices], Yc[valid_indices], Zc[valid_indices]

        rospy.loginfo(f"Number of valid points after filtering: {len(Xc)}")

        x_proj = (fx * Xc / Zc + cx).astype(int)
        y_proj = (fy * Yc / Zc + cy).astype(int)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Use vectorized operations instead of loops
        valid_points = (x_proj >= 0) & (x_proj < image.shape[1]) & (y_proj >= 0) & (y_proj < image.shape[0])
        x_proj, y_proj = x_proj[valid_points], y_proj[valid_points]

        rospy.loginfo(f"Number of valid projected points on the image: {len(x_proj)}")

        image[y_proj, x_proj] = [0, 0, 255]  # Red color
        mask[y_proj, x_proj] = 255

        projected_image_msg = bridge.cv2_to_imgmsg(image, encoding="bgr8")
        mask_msg = bridge.cv2_to_imgmsg(mask, encoding="mono8")

        projection_publisher.publish(projected_image_msg)
        mask_publisher.publish(mask_msg)

        rospy.loginfo("Published projected image and mask.")

                # Publish the colorized point cloud
        colored_points = colorize_point_cloud(lidar_points, image, transformed_points)
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("r", 12, PointField.FLOAT32, 1),
            PointField("g", 16, PointField.FLOAT32, 1),
            PointField("b", 20, PointField.FLOAT32, 1),
        ]
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "velodyne_VLP32"

        colorized_pc2 = pc2.create_cloud(header, fields, colored_points)
        colorized_publisher.publish(colorized_pc2)

    except Exception as e:
        rospy.logerr(f"Error in project_points: {e}")

class LidarCameraProjection:
    def __init__(self):
        rospy.init_node("lidar_camera_projection_node")

        self.lidar_topic = rospy.get_param("~lidar_topic", "/VLP32/velodyne_points")
        self.camera_topic_base = rospy.get_param("~camera_topic_base", "/zed2i/zed_node/left/image_rect_color")
        self.use_compressed = rospy.get_param("~use_compressed", True)
        self.output_topic = rospy.get_param("~output_topic", "/projected_points_image")
        self.mask_topic = rospy.get_param("~mask_topic", "/projected_points_mask")
        self.colorized_topic = rospy.get_param("~colorized_topic", "/colorized_pointcloud")

        self.time_tolerance = rospy.get_param("~time_tolerance", 0.06)
        self.processing_rate = rospy.get_param("~processing_rate", 10)  # Hz
        self.skip_frames = rospy.get_param("~skip_frames", 3)
        self.colorized_publisher = rospy.Publisher(self.colorized_topic, PointCloud2, queue_size=1)


        rospy.loginfo("Initializing publishers and subscribers.")

        self.projection_publisher = rospy.Publisher(self.output_topic, Image, queue_size=1)
        self.mask_publisher = rospy.Publisher(self.mask_topic, Image, queue_size=1)

        # Set up the correct image subscriber based on the compression flag
        if self.use_compressed:
            self.camera_topic = self.camera_topic_base + "/compressed"
            self.image_sub = message_filters.Subscriber(self.camera_topic, CompressedImage)
        else:
            self.camera_topic = self.camera_topic_base
            self.image_sub = message_filters.Subscriber(self.camera_topic, Image)

        self.lidar_sub = message_filters.Subscriber(self.lidar_topic, PointCloud2)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub],
            queue_size=5,
            slop=self.time_tolerance
        )
        self.ts.registerCallback(self.callback)

        self.lock = Lock()
        self.latest_data = None
        self.frame_count = 0

        rospy.Timer(rospy.Duration(1.0 / self.processing_rate), self.process_data)

        rospy.loginfo("LidarCameraProjection node initialized.")

    def callback(self, image_msg, lidar_msg):
        with self.lock:
            self.latest_data = (image_msg, lidar_msg)
        rospy.loginfo("Received synchronized LiDAR and image messages.")

    def process_data(self, event):
        with self.lock:
            if self.latest_data is None:
                rospy.loginfo("No synchronized data available for processing.")
                return
            image_msg, lidar_msg = self.latest_data
            self.latest_data = None

        self.frame_count += 1

        if self.frame_count % self.skip_frames != 0:
            rospy.loginfo(f"Skipping frame {self.frame_count}")
            return
        
        try:
            rospy.loginfo("Processing synchronized data.")

            # Handle image based on compression flag
            if self.use_compressed:
                np_arr = np.frombuffer(image_msg.data, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
            else:
                image = bridge.imgmsg_to_cv2(image_msg, "bgr8")

            image = cv2.resize(image, (image.shape[1]*2, image.shape[0]*2))
            lidar_points = np.array(list(pc2.read_points(lidar_msg, field_names=("x", "y", "z"), skip_nans=True)))
            rospy.loginfo("********************")

            rospy.loginfo(f"Image timestamp: {image_msg.header.stamp.to_sec()}")
            rospy.loginfo(f"LiDAR timestamp: {lidar_msg.header.stamp.to_sec()}")
            rospy.loginfo(f"Time difference: {abs(image_msg.header.stamp.to_sec() - lidar_msg.header.stamp.to_sec())}")
            rospy.loginfo(f"LiDAR points count: {lidar_points.shape[0]}")
            rospy.loginfo(f"Image dimensions: {image.shape}")

            project_points(lidar_points, image, self.projection_publisher, self.mask_publisher, self.colorized_publisher)

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
