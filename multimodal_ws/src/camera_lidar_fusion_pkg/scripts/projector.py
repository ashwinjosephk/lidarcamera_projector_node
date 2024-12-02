#!/usr/bin/env python3


import cv2
import yaml
import rospkg 
import os

import rospy
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import message_filters
import numpy as np
from matplotlib import cm
import pcl
import cv2




def load_config(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)



class LiDARCameraProjectionNode:
    def __init__(self, config):
        self.bridge = CvBridge()
        rospy.init_node('lidar_camera_projection_node')

        # Load config parameters
        self.camera_topic = config['camera_topic']
        self.lidar_topic = config['lidar_topic']
        self.output_topic = config['output_topic']

        self.cam_to_cam_calibration_file = config['cam_to_cam_calibration_file']
        self.imu_to_velo_calibration_file = config['imu_to_velo_calibration_file']   
        self.velo_to_cam_calibration_file = config['velo_to_cam_calibration_file']

        # Publishers and Subscribers
        self.pub = rospy.Publisher(self.output_topic, Image, queue_size=10)
        self.camera_sub = message_filters.Subscriber(self.camera_topic, Image)
        self.lidar_sub = message_filters.Subscriber(self.lidar_topic, PointCloud2)

        

        self.T_ref0_ref2 = None
        self.T_velo_ref0 = None
        self.T_imu_velo = None
        self.T_velo_cam2 = None
        self.T_cam2_velo = None
        self.T_imu_cam2 = None
        self.T_cam2_imu= None



        self.rainbow_r = cm.get_cmap('plasma', lut=100)
        self.get_color = lambda z : [255*val for val in self.rainbow_r(int(z.round()))[:3]]


        self.load_calibration()
        print("*****************")
        print("self.T_ref0_ref2",self.T_ref0_ref2)
        print("self.T_velo_ref0",self.T_velo_ref0)
        print("self.T_imu_velo",self.T_imu_velo)
        print("self.T_velo_cam2",self.T_velo_cam2)
        print("self.T_cam2_velo",self.T_cam2_velo)
        print("self.T_imu_cam2",self.T_imu_cam2)
        print("self.T_cam2_imu",self.T_cam2_imu)
        print("*****************")
        print("Node initialized successfully.")

        print(f"Camera Topic: {self.camera_topic}")

        print(f"Lidar Topic: {self.lidar_topic}")

        print(f"Output Topic: {self.output_topic}")
        print("*****************")

        # Initialize video writer
        self.record_video = False  # Default to False
        self.video_writer = None

        self.video_output_file = config.get('video_output_file', 'projected_points_video.mp4')

        self.video_fps = config.get('video_fps', 10)  # Adjust FPS as needed

        


        ts = message_filters.ApproximateTimeSynchronizer(
            [self.camera_sub, self.lidar_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.callback)



    def lidar_msg_to_points(self, lidar_msg):
        # Extract points from PointCloud2 message

        point_step = lidar_msg.point_step

        data = lidar_msg.data

        num_points = len(data) // point_step

        fields = lidar_msg.fields

        

        # Assuming we have x, y, z fields

        x_offset = next((field.offset for field in fields if field.name == 'x'), None)

        y_offset = next((field.offset for field in fields if field.name == 'y'), None)

        z_offset = next((field.offset for field in fields if field.name == 'z'), None)

        

        if x_offset is None or y_offset is None or z_offset is None:

            rospy.logerror("Could not find x, y, z fields in PointCloud2 message.")

            return

        

        lidar_points = np.zeros((num_points, 3), dtype=np.float32)

        for i in range(num_points):

            point_data = data[i * point_step:(i + 1) * point_step]

            lidar_points[i, 0] = np.frombuffer(point_data[x_offset:x_offset + 4], dtype=np.float32)[0]

            lidar_points[i, 1] = np.frombuffer(point_data[y_offset:y_offset + 4], dtype=np.float32)[0]

            lidar_points[i, 2] = np.frombuffer(point_data[z_offset:z_offset + 4], dtype=np.float32)[0]

        return lidar_points

    def callback(self, camera_msg, lidar_msg):

        # Convert ROS Image message to OpenCV image

        bgr_image = self.bridge.imgmsg_to_cv2(camera_msg, desired_encoding='bgr8')
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # print("rgb_image.shape",rgb_image.shape)
        



        lidar_points = self.lidar_msg_to_points(lidar_msg)

        velo_uvz = self.project_velobin2uvz(lidar_points, self.T_velo_cam2, rgb_image)

        

        # Draw projected points on image

        projected_image = self.draw_velo_on_image(velo_uvz, rgb_image.copy())
        # print("projected_image.shape",projected_image.shape)
        

        # Publish projected image as ROS Image message

        output_msg = self.bridge.cv2_to_imgmsg(projected_image, encoding='bgr8')

        output_msg.header = camera_msg.header  # Copy header from original image message

        

        self.pub.publish(output_msg)

       




        
    def load_calibration(self):

        with open(self.cam_to_cam_calibration_file, 'r') as f:

            calib = f.readlines()

        # get projection matrices (rectified left camera --> left camera (u,v,z))
        P_rect2_cam2 = np.array([float(x) for x in calib[25].strip().split(' ')[1:]]).reshape((3, 4))

        # get rectified rotation matrices (left camera --> rectified left camera)
        R_ref0_rect2 = np.array([float(x) for x in calib[24].strip().split(' ')[1:]]).reshape((3, 3))


        # add (0,0,0) translation and convert to homogeneous coordinates
        R_ref0_rect2 = np.insert(R_ref0_rect2, 3, values=[0,0,0], axis=0)

        R_ref0_rect2 = np.insert(R_ref0_rect2, 3, values=[0,0,0,1], axis=1)

        # get rigid transformation from Camera 0 (ref) to Camera 2
        R_2 = np.array([float(x) for x in calib[21].strip().split(' ')[1:]]).reshape((3, 3))

        t_2 = np.array([float(x) for x in calib[22].strip().split(' ')[1:]]).reshape((3, 1))

        # get cam0 to cam2 rigid body transformation in homogeneous coordinates
        self.T_ref0_ref2 = np.insert(np.hstack((R_2, t_2)), 3, values=[0,0,0,1], axis=0)




        #Load LiDAR and GPS/IMU Calibration Data
        self.T_velo_ref0 = self.get_rigid_transformation(self.velo_to_cam_calibration_file)

        self.T_imu_velo = self.get_rigid_transformation(self.imu_to_velo_calibration_file)

        # transform from velo (LiDAR) to left color camera (shape 3x4)
        self.T_velo_cam2 = P_rect2_cam2 @ R_ref0_rect2 @ self.T_ref0_ref2 @ self.T_velo_ref0 

        # homogeneous transform from left color camera to velo (LiDAR) (shape: 4x4)
        self.T_cam2_velo = np.linalg.inv(np.insert(self.T_velo_cam2, 3, values=[0,0,0,1], axis=0)) 

        # transform from IMU to left color camera (shape 3x4)
        self.T_imu_cam2 = self.T_velo_cam2 @ self.T_imu_velo

        # homogeneous transform from left color camera to IMU (shape: 4x4)
        self.T_cam2_imu = np.linalg.inv(np.insert(self.T_imu_cam2, 3, values=[0,0,0,1], axis=0)) 




    def get_rigid_transformation(self, calib_path):
        ''' Obtains rigid transformation matrix in homogeneous coordinates (combination of
            rotation and translation.
            Used to obtain:
                - LiDAR to camera reference transformation matrix 
                - IMU to LiDAR reference transformation matrix
            '''
        with open(calib_path, 'r') as f:
            calib = f.readlines()

        R = np.array([float(x) for x in calib[1].strip().split(' ')[1:]]).reshape((3, 3))
        t = np.array([float(x) for x in calib[2].strip().split(' ')[1:]])[:, None]

        T = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))
        
        return T

    def draw_velo_on_image(self, velo_uvz, image):
   
        # unpack LiDAR points
        u, v, z = velo_uvz

        color_map=self.get_color

        # draw LiDAR point cloud on blank image
        for i in range(len(u)):
            cv2.circle(image, (int(u[i]), int(v[i])), 1, 
                    color_map(z[i]), -1);

        return image
    
    # get LiDAR points and transform them to image/camera space
    #self.T_velo_cam2
    def project_velobin2uvz(self, lidar_points, T_velo_cam2, image, remove_plane=True):

        ''' Projects LiDAR point cloud onto the image coordinate frame (u, v, z)

            '''

        # Convert lidar points to homogeneous coordinates

        xyzw = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))

        # Project velo (x, y, z) onto camera (u, v, z) coordinates

        velo_uvz = self.xyzw2camera(xyzw.T, T_velo_cam2, image=image)

        return velo_uvz
    
    def xyzw2camera(self, xyz, T, image=None, remove_outliers=True):
        ''' maps xyxw homogeneous points to camera (u,v,z) space. The xyz points can 
            either be velo/LiDAR or GPS/IMU, the difference will be marked by the 
            transformation matrix T.
            '''
        # convert to (left) camera coordinates
        camera =  T @ xyz

        # delete negative camera points
        camera  = np.delete(camera , np.where(camera [2,:] < 0)[0], axis=1) 

        # get camera coordinates u,v,z
        camera[:2] /= camera[2, :]

        # remove outliers (points outside of the image frame)
        if remove_outliers:
            u, v, z = camera
            img_h, img_w, _ = image.shape
            u_out = np.logical_or(u < 0, u > img_w)
            v_out = np.logical_or(v < 0, v > img_h)
            outlier = np.logical_or(u_out, v_out)
            camera = np.delete(camera, np.where(outlier), axis=1)

        return camera
    
    def __del__(self):

        if self.video_writer is not None:

            self.video_writer.release()

if __name__ == "__main__":

    config = load_config('/home/knadmin/Ashwin/ros_projects/multimodal_ws/src/camera_lidar_fusion_pkg/config/kitti.yaml')
    print("Loaded Configuration:")

    print(config)
    node = LiDARCameraProjectionNode(config)

    

    try:

        rospy.spin()

    except KeyboardInterrupt:

        print("Node stopped by user.")
