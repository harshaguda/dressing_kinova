#!/usr/bin/env python3

from stable_baselines3 import PPO
import torch

from delta_pose_control import DeltaPoseControl
from utils import CustomCombinedExtractor, DummyEnv

dpc = DeltaPoseControl()
dummy_env = DummyEnv()
### Add code to use policy model

model = PPO.load("model.zip", env=dummy_env, 
         custom_objects={"policy_kwargs":
                         dict(
                             features_extractor_class=CustomCombinedExtractor,
                             normalize_images=False,
                             activation_fn=torch.nn.modules.activation.Tanh
                         )}
         )

dpc.set_cartesian_pose(0.1, 0, 0)
dpc.set_cartesian_pose(0, 0.1, 0) 
dpc.set_cartesian_pose(0, 0, -0.1)