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
import argparse
import os
import time

args = argparse.ArgumentParser()
args.add_argument('--camid', type=int, default=4, help='Camera ID for video capture')
args.add_argument('--record', action='store_true', help='Record images')
args = args.parse_args()

camid = 4
emo = Emotions(device='cuda' if torch.cuda.is_available() else 'cpu', camid=camid)
action_rec = ActionsPerf(device='cuda' if torch.cuda.is_available() else 'cpu', camid=camid)

# tc = TrajectoryControl(home=[0.3, -0.3, 0.505])

poses = MediaPipe3DPose(debug=True, translate=True)
dmp = DMPDG(n_dmps=3, n_bfs=500, T=1.5, dt=0.01, tau=1.0, tau_y=1.0, pattern="discrete", dmp_type="vanilla")
is_Approached = False
is_Extended = False
i = 0

if args.record:
    time_str = time.strftime("%Y%m%d-%H%M%S")
    path = f"paper/{time_str}"
    if not os.path.exists(path):
        os.makedirs(path)
    
image_i = 0

arm_poses = []
for i in range(50):
    arm_pos, image = poses.get_arm_points()
    arm_poses.append(arm_pos)
    cv2.imshow("pose", image)
    if args.record:
        cv2.imwrite(f"{path}/pose_{image_i}.png", image)
        image_i += 1
    cv2.waitKey(1)

arm_poses = np.array(arm_poses)
arm_poses = arm_poses[25:]#.mean(axis=0)

arm_pos= arm_poses[(arm_poses.sum(axis=-1) != 0.).sum(axis=-1) >= 1].mean(axis=0)

wrist = arm_pos[0]
elbow = arm_pos[1]

v_n = elbow - wrist
v_n /= np.linalg.norm(v_n)
ext_wrist = -v_n * 0.15 + wrist

init_traj = np.vstack((ext_wrist, arm_pos))
np.savetxt(f"{path}/init_traj.txt", init_traj, delimiter=",")
# y_des = make_arm_trajectory(arm_poses[25:].mean(axis=0))

y_des = make_arm_trajectory(arm_pos)
dmp.imitate_trajectory(y_des)
traj, _, _ = dmp.rollout()
traj = traj[::10].copy()

# plt.plot(traj[:, 0], traj[:, 1])
## clip values to avoid going out of workspace
traj = traj.clip(min=[0.25, -0.5, 0.1], max=[0.9, 0.5, 0.7]).copy()
plt.plot(traj[:, 0], traj[:, 1])
plt.text(traj[0,0], traj[0,1], "Start")
plt.text(traj[-1,0], traj[-1,1], "Goal")
plt.xlim(0, 1)
plt.ylim(-0.6, 0.6)
plt.show()

# traj = traj.clip(-0.5, 0.5)
if args.record:
    np.savetxt(f"{path}/trajectory.txt", traj, delimiter=',')
# success = tc.example_cartesian_waypoint_action(traj)
for tp in traj:
    # for i in range(20):
    #     e_image, emotion, engagement = emo.predict_emotions()
    #     cv2.imshow('Emotion Detection', e_image)
    #     cv2.waitKey(1)
    emotion = ""
        
    # while emotion not in ["Neutral", "Happiness"]:
    #     e_image, emotion, engagement = emo.predict_emotions()
    #     print("inloop")
    #     cv2.imshow("emotion", e_image)
    #     cv2.waitKey(1)
    for i in range(5):
        arm_pos, image = poses.get_arm_points()
        cv2.imshow("pose", image)
        if args.record:
            cv2.imwrite(f"{path}/pose_{image_i}.png", image)
            image_i += 1
        cv2.waitKey(1)
    # print(arm_pos)
    print("Dressing", emotion)
    if arm_pos[-1].sum() != 0.0:
        traj[-1] = arm_pos[-1]
        
    print(tp)
    pc.set_pose(tp[0], tp[1], tp[2])
