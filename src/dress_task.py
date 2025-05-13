#!/usr/bin/env python3

from stable_baselines3 import PPO
import torch
import cv2

from delta_pose_control import DeltaPoseControl
from utils import CustomCombinedExtractor, DummyEnv
import numpy as np
import rospy

from pose_estimation import MediaPipe3DPose

def process_image(frame):
    # frame = cv2.rotate(frame, cv2.ROTATE_180)
    resized_image = cv2.resize(frame, (214, 214))
    image = np.swapaxes(resized_image, 0, 2)/255.
    return np.expand_dims(image, axis=0), resized_image

dpc = DeltaPoseControl()
poses = MediaPipe3DPose(debug=True)
dummy_env = DummyEnv()
### Add code to use policy model

model = PPO.load("/home/userlab/iri_lab/iri_ws/src/dressing_kinova/models/best_model.zip", env=dummy_env, 
         custom_objects={"policy_kwargs":
                         dict(
                             features_extractor_class=CustomCombinedExtractor,
                             normalize_images=False,
                             activation_fn=torch.nn.modules.activation.Tanh
                         )}
         )
action_factor = 0.025

obs = poses.get_arm_positions()
base_action = torch.zeros(1, 3)
d = rospy.Duration(0.5)
for i in range(2048):

    action, _ = model.predict(obs)
    x, y, z = -action[0]*action_factor
    base_action += torch.tensor([x, y, z]).reshape(1,3)
    dpc.set_cartesian_pose(x, y, z)
    # cv2.imshow('frame', res_img)
    cv2.waitKey(1)
    rospy.sleep(d)
