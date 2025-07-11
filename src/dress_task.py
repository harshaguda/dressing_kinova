#!/usr/bin/env python3

from stable_baselines3 import PPO
import torch
import cv2

from delta_pose_control import DeltaPoseControl
from utils import CustomCombinedExtractor, DummyEnv
import numpy as np
import rospy
from pose_estimation import MediaPipe3DPose
from controller import ControllerDressing
from dmp_kinova import DMPDG, make_arm_trajectory

def process_image(frame):
    # frame = cv2.rotate(frame, cv2.ROTATE_180)
    resized_image = cv2.resize(frame, (214, 214))
    image = np.swapaxes(resized_image, 0, 2)/255.
    return np.expand_dims(image, axis=0), resized_image

dpc = DeltaPoseControl()
rospy.sleep(5)
poses = MediaPipe3DPose(debug=True, translate=True)
dummy_env = DummyEnv(obs="pos")
dmp = DMPDG(n_dmps=3, n_bfs=500, T=1.5, dt=0.01, tau=1.0, tau_y=1.0, pattern="discrete", dmp_type="vanilla")
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
dmp_flag = False
for i in range(2048):
    ee_pos, rot = dpc.get_ee_pose()
    arm_pos, image = poses.get_arm_points()
    
    cv2.imwrite(f"/home/userlab/iri_lab/iri_ws/src/dressing_kinova/recordings/{i}.jpg", image)
    ee_pos[2] -= 0.12
    
    arm_to_ee = np.linalg.norm(arm_pos[0] - ee_pos)
    # dpc.set_cartesian_posed(0, 0.1, 0)
    print(arm_to_ee)
    if (arm_to_ee < 0.8):
        t = 0.0
        dt = 0.01
        y_des = make_arm_trajectory(arm_pos)
        dmp.imitate_trajectory(y_des)
        dmp.rollout()
        while (t < 1.5):
            print("Moving arm")
            arm_pos_des, _, _ = dmp.step()
            t += dt
            ee_pos, rot = dpc.get_ee_pose()
            # diff = arm_pos_des - ee_pos
            # print(arm_pos[-1] - ee_pos)
            # print(diff)
            # print(arm_pos_des)
            x, y, z = arm_pos_des
            print(x, y, z)
            dpc.set_cartesian_posed(x, y, z)

            cv2.waitKey(1)
            rospy.sleep(d)

    rospy.on_shutdown(dpc.shutdown_node)
    rospy.sleep(d)