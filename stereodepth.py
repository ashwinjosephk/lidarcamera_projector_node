import cv2
import numpy as np

def compute_depth_from_stereo(left_image, right_image, camera_intrinsics):
    """
    Compute a depth image from stereo images using OpenCV's StereoSGBM.
    :param left_image: Left rectified image (BGR or grayscale)
    :param right_image: Right rectified image (BGR or grayscale)
    :param camera_intrinsics: Dictionary with 'fx' (focal length in pixels)
    :return: Depth image in meters (same size as input images)
    """
    baseline = 0.12  # Baseline of ZED2i stereo camera in meters

    # Convert images to grayscale if needed
    if len(left_image.shape) == 3:
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_image
        right_gray = right_image

    # Stereo matcher settings
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,  # Must be a multiple of 16
        blockSize=9,
        P1=8 * 3 * 9 ** 2,
        P2=32 * 3 * 9 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    # Compute disparity
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0  # Normalize

    # Avoid division by zero (replace invalid disparity values with small nonzero value)
    disparity[disparity <= 0] = 0.1

    # Compute depth using Depth = (fx * B) / disparity
    depth = (camera_intrinsics["fx"] * baseline) / disparity

    return depth
