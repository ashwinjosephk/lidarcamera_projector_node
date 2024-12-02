#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import yaml
import rospkg 
import os

def load_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        rospy.logerr(f"Failed to load image at path: {image_path}")
        return None
    print(f"Loaded image shape: {frame.shape}, dtype: {frame.dtype}")
    return frame


def publish_message():
    # Load configuration from YAML file in the package config directory
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('example_pkg')  # Replace with your package name
    config_file_path = os.path.join(package_path, 'config', 'project_config.yaml')

    # Load the configuration file
    with open(config_file_path, 'r') as stream:
        config = yaml.safe_load(stream)

    # Construct full path to the image based on the configuration
    image_path = os.path.join(package_path, config['image_path'])
    
    # Load the image
    frame = load_image(image_path)

    if frame is None:
        rospy.logerr("Could not load image at path: {}".format(image_path))
        return

    # Node is publishing to the topic specified in the configuration
    topic_name = config.get('topic_name', 'video_frames')  # Default topic name if not specified
    pub = rospy.Publisher(topic_name, Image, queue_size=10)
    
    # Initialize the ROS node
    rospy.init_node('image_pub_py', anonymous=True)
    
    # Set the publishing rate
    rate = rospy.Rate(10)  # 10 Hz

    # Used to convert between ROS and OpenCV images
    br = CvBridge()

    while not rospy.is_shutdown():
        # Print debugging information to the terminal
        rospy.loginfo('Publishing image frame on topic: {}'.format(topic_name))

        try:
            # Ensure the frame is in BGR format before publishing
            if len(frame.shape) == 2:  # If it's a grayscale image (single channel)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert to BGR
            
            # Publish the image.
            pub.publish(br.cv2_to_imgmsg(frame, encoding="bgr8"))
        
        except Exception as e:
            rospy.logerr("Error publishing image: {}".format(e))

        # Sleep just enough to maintain the desired rate
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_message()
    except rospy.ROSInterruptException:
        pass
