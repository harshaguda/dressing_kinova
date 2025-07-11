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


dpc = DeltaPoseControl(home=[0.3, -0.3, 0.505])
rospy.sleep(5)