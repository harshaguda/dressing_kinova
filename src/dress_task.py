#!/usr/bin/env python3

from stable_baselines3 import PPO
import torch
import cv2

from delta_pose_control import DeltaPoseControl
from utils import CustomCombinedExtractor, DummyEnv
import numpy as np
import rospy
import time
from pose_estimation import MediaPipe3DPose
from controller import ControllerDressing

def process_image(frame):
    # frame = cv2.rotate(frame, cv2.ROTATE_180)
    resized_image = cv2.resize(frame, (214, 214))
    image = np.swapaxes(resized_image, 0, 2)/255.
    return np.expand_dims(image, axis=0), resized_image

dpc = DeltaPoseControl()
rospy.sleep(5)
poses = MediaPipe3DPose(debug=True, translate=True)
dummy_env = DummyEnv(obs="pos")
controller = ControllerDressing()
### Add code to use policy model

model = PPO.load("/home/userlab/iri_lab/iri_ws/src/dressing_kinova/models/best_model.zip", env=dummy_env, 
         # custom_objects={"policy_kwargs":
         #                 dict(
         #                     features_extractor_class=CustomCombinedExtractor,
         #                     normalize_images=False,
         #                     activation_fn=torch.nn.modules.activation.Tanh
         #                 )}
         )
action_factor = 0.025

base_action = torch.zeros(1, 3)
d = rospy.Duration(0.5)
ee_pos_aligned = np.zeros(3)
def simple_delta_controller(a, b):
    action = a - b
    return action / np.linalg.norm(action)

def angle(elbow, wrist):
    """
    Calculate the angle between the line segment from elbow to wrist and the x-axis."""
    if elbow[0] == wrist[0]:  # Avoid division by zero
        return 0
    if elbow[1] == wrist[1]:  # Horizontal line
        return np.pi / 2
    if elbow[0] == wrist[0] and elbow[1] == wrist[1]:  # Same point
        return 0
    if np.sum(elbow) == 0 or np.sum(wrist) == 0:  # Avoid zero vectors
        return 0

        
    m = (wrist[1] - elbow[1]) / (elbow[0] - wrist[0] )
    return np.arctan(m)

for i in range(2048):
    ee_pos, rot = dpc.get_ee_pose()
    ## ee_pos is y, x, z of the chosen axis.
    # print(ee_pos.shape, ee_pos)
    # ee_pos_aligned[1], ee_pos_aligned[0], ee_pos_aligned[2] = ee_pos[0], ee_pos[1], ee_pos[2]
    # arm_pos = np.array([[0.10959604, 0.18741445,  1.00900006],
    #                      [0.32794195, 0.22820899,  1.37300003],
    #                      [0.39954785, 0.30343255,  1.49000013]])
    arm_pos, image = poses.get_arm_points()
    # sth = poses.get_arm_points()
    # print(sth)
    # exit()
    cv2.imwrite(f"/home/userlab/iri_lab/iri_ws/src/dressing_kinova/recordings/{i}.jpg", image)
    ee_pos[2] -= 0.12
    
   
    # print("ee_pos", ee_pos)
    # print("arm_pos", arm_pos)
    arm_pos[0] = arm_pos[0] - [0, 0.4, 0] 
    action, _ = controller.meta_action(arm_pos, ee_pos)
    tx = angle(arm_pos[1], arm_pos[0])
    # action = simple_delta_controller(arm_pos[0], ee_pos)
    # print("action", action)
    print(ee_pos, arm_pos[0])
    # print(action.shape)
    x, y, z = action * action_factor
    dpc.set_cartesian_pose(x=0, y=0, z=0, tx=tx)
    cv2.waitKey(1)
    
    # k = cv2.waitKey(1)
    # if k==27:    # Esc key to stop
    #     exit()
    rospy.on_shutdown(dpc.shutdown_node)
    rospy.sleep(d)