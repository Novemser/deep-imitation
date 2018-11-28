# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
import time
from sys import platform
import matplotlib.pyplot as plt
import math

# Remember to add your installation path here

# If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
if '2.7' in sys.version:
    sys.path.append('/usr/local/python/')
else:
    sys.path.append('/usr/local/python/openpose')

# Parameters for OpenPose. Take a look at C++ OpenPose example for meaning of components. Ensure all below are filled
try:
    from openpose import *
except:
    raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
os.chdir(os.path.dirname(os.path.realpath(__file__)))
keypoint_misVal = 8
init_img = cv2.imread('../data/girl1_1_danceinit.png')
# last_create_label = np.zeros((336, 336))
params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368"
params["model_pose"] = "BODY_25"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.05
# If GPU version is built, and multiple GPUs are available, set the ID here
params["num_gpu_start"] = 0
params["disable_blending"] = False
# Ensure you point to the correct path where models are located
params["default_model_folder"] =  "../model/"
# Construct OpenPose object allocates GPU memory

body25_joint_info = [[17,15],
[15,0],
[0,16],
[16,18],
[0,1],
[1,2],
[2,3],
[3,4],
[1,5],
[5,6],
[6,7],
[1,8],
[8,9],
[8,12],
[9,10],
[10,11],
[11,22],
[22,23],[11,24],[12,13],[13,14],[14,21],[14, 19],[19,20]]

coco_joint_info = [[1, 2],
[1, 5],
[2, 3],
[3, 4],
[5, 6],
[6, 7],
[1, 8],
[8, 9],
[ 9, 10],
[ 1, 11],
[11, 12],
[12, 13],
[1, 0],
[ 0, 14],
[14, 16],
[ 0, 15],
[15, 17]]

def create_label(shape, joint_list):
    label = np.zeros(shape, dtype=np.uint8)
    miss_kpNum = 0
    if params["model_pose"] == 'COCO':
        total_joint = 17
        joint_info = coco_joint_info
    elif params["model_pose"] == 'BODY_25':
        total_joint = 24
        joint_info = body25_joint_info

    for limb_type in range(total_joint):
        joint_indices = joint_info[limb_type]
        joint_coords = joint_list[joint_indices, :2]
        if 0 in joint_coords:
            miss_kpNum += 1
            if miss_kpNum >= keypoint_misVal:
                return init_label
                #return last_create_label
            else: continue
        coords_center = tuple(np.round(np.mean(joint_coords, 0)).astype(int))
        limb_dir = joint_coords[0, :] - joint_coords[1, :]
        limb_length = np.linalg.norm(limb_dir)
        angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))
        polygon = cv2.ellipse2Poly(coords_center, (int(limb_length / 2), 4), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(label, polygon, limb_type+1)
    return label

openpose = OpenPose(params)

def poseEstimate(img, is_square=True, resize=(480, 480)):
    """
    Param
    img: image of OpenCV format (H X W X BGR)
    is_square: if True, cut the image to a squre
    resize: if resize != 0, resize the image

    Return
    label: the pose of the target, gray image of the same resized size
    """

    if is_square is True:
        shape_dst = np.min(img.shape[:2])

        oh = (img.shape[0] - shape_dst) // 2
        ow = (img.shape[1] - shape_dst) // 2
        # Output keypoints and the image with the human skeleton blended on it

        img = img[oh:oh+shape_dst, ow:ow+shape_dst]
    img = cv2.resize(img, resize)
    keypoints, output_image = openpose.forward(img, True)
    keypoints = keypoints[0].reshape(-1, 3)
    label = create_label(img.shape[:2], keypoints)
    # last_create_label = label

    return label

init_label = poseEstimate(init_img)
if __name__ == '__main__':
    test_img = cv2.imread('../data/posetest.png')
    label = poseEstimate(test_img)
    cv2.imshow('test', label)
    cv2.waitKey(0)
