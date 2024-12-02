cmake_minimum_required(VERSION 3.0.2)
project(example_pkg)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
)
find_package(OpenCV REQUIRED)
catkin_package()

catkin_install_python(PROGRAMS
  scripts/webcam_pub.py
  scripts/webcam_sub.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

include_directories(${catkin_INCLUDE_DIRS})

