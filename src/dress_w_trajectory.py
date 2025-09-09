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
from geometry_msgs.msg import Pose, PoseArray
import tf

# from delta_pose_control import DeltaPoseControl

# dpc = DeltaPoseControl(home=[0.3, -0.3, 0.505])

from pose_control import ExampleCartesianActionsWithNotifications
pc = ExampleCartesianActionsWithNotifications()
success = pc.main()

args = argparse.ArgumentParser()
args.add_argument('--camid', type=int, default=4, help='Camera ID for video capture')
args.add_argument('--record', action='store_true', help='Record images')
args = args.parse_args()

camid = 11
emo = Emotions(device='cuda' if torch.cuda.is_available() else 'cpu', camid=camid)
action_rec = ActionsPerf(device='cuda' if torch.cuda.is_available() else 'cpu', camid=camid)

# tc = TrajectoryControl(home=[0.3, -0.3, 0.505])

pa_publisher = rospy.Publisher('/desired_trajectory', PoseArray, queue_size=10)
arm_publisher = rospy.Publisher('/detected_arm', PoseArray, queue_size=10)
listener = tf.TransformListener()
listener.waitForTransform('/base_link', '/tool_frame', rospy.Time(), rospy.Duration(4.0))
# print(trans)
# exit()
poses = MediaPipe3DPose(debug=True, translate=True)
dmp = DMPDG(n_dmps=3, n_bfs=500, T=1.5, dt=0.1, tau=1.0, tau_y=1.0, pattern="discrete", dmp_type="vanilla")
is_Approached = False
is_Extended = False
i = 0

if args.record:
    time_str = time.strftime("%Y%m%d-%H%M%S")
    path = f"paper/{time_str}"
    if not os.path.exists(path):
        os.makedirs(path)
    
image_i = 0
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
#         if args.record:
#             cv2.imwrite(f"{path}/image_{image_i}.png", new_image)
#             image_i += 1
#     if is_Extended and is_Approached:
#         print("Dresss")
#         break
    
# cv2.waitKey(5000)

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

rarm_pos = PoseArray()
for tp in arm_pos:
    ra = Pose()
    
    ra.position.x = tp[0]
    ra.position.y = tp[1]
    ra.position.z = tp[2]
    rarm_pos.poses.append(ra)
    
rarm_pos.header.frame_id = "base_link"
arm_publisher.publish(rarm_pos)

wrist = arm_pos[0]
elbow = arm_pos[1]

v_n = elbow - wrist
v_n /= np.linalg.norm(v_n)
ext_wrist = -v_n * 0.25 + wrist

# init_traj = np.vstack((ext_wrist, arm_pos))
# if args.record:
#     np.savetxt(f"{path}/init_traj.txt", init_traj, delimiter=",")
# y_des = make_arm_trajectory(arm_poses[25:].mean(axis=0))

# current_pos = moveit.get_cartesian_pose()
# ee_pos = np.array([current_pos.position.x, current_pos.position.y, current_pos.position.z])
# ee_pos = np.array([0.08, -0.59, 0.229])
(trans,rot) = listener.lookupTransform('/base_link', '/tool_frame', rospy.Time(0))
init_traj = np.vstack((np.array(trans), ext_wrist, arm_pos))
# print(init_traj.shape)
# exit()
y_des = make_arm_trajectory(init_traj)
dmp.imitate_trajectory(y_des)
traj, _, _ = dmp.rollout()


pa = PoseArray()
for tp in traj:
    p = Pose()
    
    p.position.x = tp[0]
    p.position.y = tp[1]
    p.position.z = tp[2]
    pa.poses.append(p)
    
pa.header.frame_id = "base_link"
pa_publisher.publish(pa)

# y_des = make_arm_trajectory(arm_pos)
# dmp.imitate_trajectory(y_des)
# traj, _, _ = dmp.rollout()
# traj = traj[::10].copy()
# traj = np.vstack((ext_wrist, traj))
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
for tp in traj[:-2,:]:
    # for i in range(20):
    #     e_image, emotion, engagement = emo.predict_emotions()
    #     cv2.imshow('Emotion Detection', e_image)
    #     if args.record:
    #         cv2.imwrite(f"{path}/image_{image_i}.png", e_image)
    #         image_i += 1
    #     cv2.waitKey(1)
    # # emotion = ""
        
    # while emotion not in ["Neutral", "Happiness"]:
    #     e_image, emotion, engagement = emo.predict_emotions()
    #     print("inloop")
    #     cv2.imshow("emotion", e_image)
    #     if args.record:
    #         cv2.imwrite(f"{path}/image_{image_i}.png", e_image)
    #         image_i += 1
    #     cv2.waitKey(1)
    # for i in range(5):
    #     arm_pos, image = poses.get_arm_points()
    #     cv2.imshow("pose", image)
    #     if args.record:
    #         cv2.imwrite(f"{path}/pose_{image_i}.png", image)
    #         image_i += 1
    #     cv2.waitKey(1)
    # # print(arm_pos)
    # print("Dressing", emotion)
    # if arm_pos[-1].sum() != 0.0:
    #     traj[-1] = arm_pos[-1]
    print(success)
    if success:
        print(tp)
        success = pc.set_pose(tp[0], tp[1], tp[2])
        print(success)
