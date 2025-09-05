import cv2
import numpy as np
import rospy
from pose_estimation import MediaPipe3DPose
from dmp_kinova import DMPDG, make_arm_trajectory
from full_trajectory import TrajectoryControl
from emotions import Emotions
from actions_perf import ActionsPerf
import torch
import matplotlib.pyplot as plt
# from delta_pose_control import DeltaPoseControl

# dpc = DeltaPoseControl(home=[0.3, -0.3, 0.505])

from pose_control import ExampleCartesianActionsWithNotifications
pc = ExampleCartesianActionsWithNotifications()
pc.main()

camid = 10
emo = Emotions(device='cuda' if torch.cuda.is_available() else 'cpu', camid=camid)
action_rec = ActionsPerf(device='cuda' if torch.cuda.is_available() else 'cpu', camid=camid)

# tc = TrajectoryControl(home=[0.3, -0.3, 0.505])

poses = MediaPipe3DPose(debug=True, translate=True)
dmp = DMPDG(n_dmps=3, n_bfs=500, T=1.5, dt=0.01, tau=1.0, tau_y=1.0, pattern="discrete", dmp_type="vanilla")
is_Approached = False
is_Extended = False
i = 0
# while True:
#     arm_pos, image = poses.get_arm_points()
#     e_image, emotion, engagement = emo.predict_emotions()
#     action_label, image_a = action_rec.predict_actions(e_image)
#     if (not is_Approached) and (not is_Extended) and (action_label == "Approach"):
#         is_Approached = True
#     if (is_Approached) and (action_label == "ExtendArm"):
#         is_Extended = True
#     # Display the frame with detected faces and emotions
#     if e_image is not None:
#         new_image = np.concatenate((image, image_a), axis=1)
#         cv2.imshow('Emotion Detection', new_image)
#         cv2.imwrite(f"paper/image_{i}.png", new_image)
#         i += 1
#     if is_Extended and is_Approached:
#         print("Dresss")
#         break
    
#     cv2.waitKey(1)
arm_poses = []
for i in range(50):
    arm_pos, image = poses.get_arm_points()
    arm_poses.append(arm_pos)
    cv2.imshow("pose", image)
    # cv2.imwrite(f"paper/pose_{i}.png", new_image)
    cv2.waitKey(1)

arm_poses = np.array(arm_poses)
arm_poses = arm_poses[25:]#.mean(axis=0)
# np.savetxt("arm_poses.txt", arm_poses.reshape(-1, 3), delimiter=",")
# np.save("arm_pos.npy", arm_poses)
# exit()
arm_pos= np.mean(arm_poses[(arm_poses.sum(axis=-1) != 0.).sum(axis=-1) >= 1], axis=0)
arm_pos[:, -1] += 0.1
arm_pos[:, 0] -= 0.18
# print(arm_poses, arm_poses[25:].mean(axis=0).shape)
wrist = arm_pos[0]
elbow = arm_pos[1]

v_n = elbow - wrist
v_n /= np.linalg.norm(v_n)
ext_wrist = -v_n * 0.15 + wrist

init_traj = np.vstack((ext_wrist, arm_pos))
np.savetxt("init_traj.txt", init_traj, delimiter=",")
# y_des = make_arm_trajectory(arm_poses[25:].mean(axis=0))

y_des = make_arm_trajectory(arm_pos)
dmp.imitate_trajectory(y_des)
traj, _, _ = dmp.rollout()
traj = traj[::10].copy()

plt.plot(traj[:, 0], traj[:, 1])
## clip values to avoid going out of workspace
# traj = traj.clip(min=[0.25, -0.5, 0.1], max=[0.9, 0.5, 0.7]).copy()
plt.plot(traj[:, 0], traj[:, 1])
plt.xlim(0, 1)
plt.ylim(-0.6, 0.6)
plt.show()

# traj = traj.clip(-0.5, 0.5)

np.savetxt("trajectory.txt", traj, delimiter=',')
# success = tc.example_cartesian_waypoint_action(traj)
for tp in traj:
#     # e_image, emotion, engagement = emo.predict_emotions()
#     # cv2.imshow('Emotion Detection', e_image)
#     # cv2.waitKey(1)
#     # while emotion not in ["Neutral", "Happiness"]:
#     #     e_image, emotion, engagement = emo.predict_emotions()
#     #     cv2.waitKey(1)
    print(tp)
    pc.set_pose(tp[0], tp[1], tp[2])
