#!/usr/bin/env python3

from stable_baselines3 import PPO
import torch
import cv2

from delta_pose_control import DeltaPoseControl
from utils import CustomCombinedExtractor, DummyEnv
import numpy as np
import rospy

def process_image(frame):
    resized_image = cv2.resize(frame, (214, 214))
    image = np.swapaxes(resized_image, 0, 2)
    return np.expand_dims(image, axis=0)

dpc = DeltaPoseControl()
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
obs = {}
obs['hvert'] = torch.zeros(1, 9)
obs['image'] = torch.zeros(1, 3, 214, 214)
cap = cv2.VideoCapture(4)

flag, frame = cap.read()
d = rospy.Duration(0.5)
for i in range(2048):
    _, frame = cap.read()
    obs['image'] = process_image(frame)
    action, _ = model.predict(obs)
    x, y, z = action[0]*action_factor
    dpc.set_cartesian_pose(x, y, z)
    rospy.sleep(d)
