import cv2

from delta_pose_control import DeltaPoseControl
from utils import CustomCombinedExtractor, DummyEnv
import numpy as np
import rospy
from pose_estimation import MediaPipe3DPose
from controller import ControllerDressing
from dmp_kinova import DMPDG, make_arm_trajectory
from full_trajectory import TrajectoryControl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d
from emotions import Emotions
import torch

emo = Emotions(device='cuda' if torch.cuda.is_available() else 'cpu', camid=10)
tc = TrajectoryControl(home=[0.3, -0.3, 0.505])

poses = MediaPipe3DPose(debug=True, translate=True)
dmp = DMPDG(n_dmps=3, n_bfs=500, T=1.5, dt=0.01, tau=1.0, tau_y=1.0, pattern="discrete", dmp_type="vanilla")

# fig = plt.figure()
# ax = Axes3D(fig)
# for i in range(30):
#     arm_pos, image = poses.get_arm_points()
    
#     plt.cla()
#     ax.plot(arm_pos[:, 0], arm_pos[:, 1], arm_pos[:, 2])
#     # plt.xlim([-0.5, 0.5])
#     # plt.ylim([-0.5, 0.5])
#     plt.draw()
#     plt.pause(0.1)

# arm_pos = np.array([[0.5, -0.1, 0.305], [0.2, 0.0, 0.305], [0.2, 0.3, 0.305]])
for i in range(50):
    arm_pos, image = poses.get_arm_points()
    e_image = emo.predict_emotions()
        
    # Display the frame with detected faces and emotions
    if e_image is not None:
        new_image = np.concatenate((image, e_image), axis=1)
        cv2.imshow('Emotion Detection', new_image)
        cv2.imwrite(f"img2/image_{i}.png", new_image)
    cv2.waitKey(1)
# arm_pos = arm_pos - np.array([0.0, 0.2, 0.0])
print("Doing motion generation")
y_des = make_arm_trajectory(arm_pos)
dmp.imitate_trajectory(y_des)
traj, _, _ = dmp.rollout()
traj = traj[::10].copy()
traj = traj.clip(-0.6, 0.55)

np.savetxt("trajectory.txt", traj, delimiter=',')
# # traj = np.array([[0.45, 1.32, 0.27], [0.56, 0.66, 0.40], [0.23, 1.03, 0.05]])
# # traj = np.array([[0.3,  0.20,   0.305], [0.4,  0.250,   0.305], [0.4,  .40,   0.305]])
success = tc.example_cartesian_waypoint_action(traj)
# # For testing purposes
rospy.set_param("/kortex_examples_test_results/waypoint_action_python", success)