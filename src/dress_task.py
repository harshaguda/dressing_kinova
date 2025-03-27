#!/usr/bin/env python3

from stable_baselines3 import PPO
import torch

from delta_pose_control import DeltaPoseControl
from utils import CustomCombinedExtractor, DummyEnv

dpc = DeltaPoseControl()
dummy_env = DummyEnv()
### Add code to use policy model

model = PPO.load("/home/userlab/iri_lab/iri_ws/src/dressing_kinova/models/model (6).zip", env=dummy_env, 
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
for i in range(2048):
    action, _ = model.predict(obs)
    x, y, z = action[0]*action_factor
    dpc.set_cartesian_pose(x, y, z)
