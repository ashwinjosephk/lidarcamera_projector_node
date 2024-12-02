#!/usr/bin/env python3
# Description:
# - Subscribes to real-time streaming video from a publisher node.
#
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com

# Import the necessary libraries
import rospy  # Python library for ROS
from sensor_msgs.msg import Image  # Image is the message type
from cv_bridge import CvBridge  # Package to convert between ROS and OpenCV Images
import cv2  # OpenCV library

def callback(data):
    # Used to convert between ROS and OpenCV images
    br = CvBridge()

    # Output debugging information to the terminal
    rospy.loginfo("Receiving video frame")

    try:
        # Convert ROS Image message to OpenCV image
        current_frame = br.imgmsg_to_cv2(data, desired_encoding='bgr8')
        
        # Display image
        cv2.imshow("Camera", current_frame)
        cv2.waitKey(1)  # Wait for 1 ms for displaying the image

    except Exception as e:
        rospy.logerr("Error converting image: {}".format(e))

def receive_message():
    # Tells rospy the name of the node.
    rospy.init_node('video_sub_py', anonymous=True)

    # Node is subscribing to the video_frames topic
    rospy.Subscriber('video_frames', Image, callback)

    # spin() simply keeps Python from exiting until this node is stopped
    rospy.spin()

    # Close down the video stream when done
    cv2.destroyAllWindows()

if __name__ == '__main__':
    receive_message()
