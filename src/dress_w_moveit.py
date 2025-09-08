#!/usr/bin/env python3

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
import sys
import time
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_srvs.srv import Empty
import numpy as np

class ExampleMoveItTrajectories(object):
  """ExampleMoveItTrajectories"""
  def __init__(self):

    # Initialize the node
    super(ExampleMoveItTrajectories, self).__init__()
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('example_move_it_trajectories')

    try:
      self.is_gripper_present = rospy.get_param(rospy.get_namespace() + "is_gripper_present", False)
      if self.is_gripper_present:
        gripper_joint_names = rospy.get_param(rospy.get_namespace() + "gripper_joint_names", [])
        self.gripper_joint_name = gripper_joint_names[0]
      else:
        self.gripper_joint_name = ""
      self.degrees_of_freedom = rospy.get_param(rospy.get_namespace() + "degrees_of_freedom", 7)

      # Create the MoveItInterface necessary objects
      arm_group_name = "arm"
      self.robot = moveit_commander.RobotCommander("robot_description")
      self.scene = moveit_commander.PlanningSceneInterface(ns=rospy.get_namespace())
      self.arm_group = moveit_commander.MoveGroupCommander(arm_group_name, ns=rospy.get_namespace())
      self.display_trajectory_publisher = rospy.Publisher(rospy.get_namespace() + 'move_group/display_planned_path',
                                                    moveit_msgs.msg.DisplayTrajectory,
                                                    queue_size=20)

      if self.is_gripper_present:
        gripper_group_name = "gripper"
        self.gripper_group = moveit_commander.MoveGroupCommander(gripper_group_name, ns=rospy.get_namespace())

      rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())
    except Exception as e:
      print (e)
      self.is_init_success = False
    else:
      self.is_init_success = True


  def reach_named_position(self, target):
    arm_group = self.arm_group
    
    # Going to one of those targets
    rospy.loginfo("Going to named target " + target)
    # Set the target
    arm_group.set_named_target(target)
    # Plan the trajectory
    (success_flag, trajectory_message, planning_time, error_code) = arm_group.plan()
    # Execute the trajectory and block while it's not finished
    return arm_group.execute(trajectory_message, wait=True)

  def reach_joint_angles(self, tolerance):
    arm_group = self.arm_group
    success = True

    # Get the current joint positions
    joint_positions = arm_group.get_current_joint_values()
    rospy.loginfo("Printing current joint positions before movement :")
    for p in joint_positions: rospy.loginfo(p)

    # Set the goal joint tolerance
    self.arm_group.set_goal_joint_tolerance(tolerance)

    # Set the joint target configuration
    if self.degrees_of_freedom == 7:
      print(pi)
      joint_positions[0] = np.deg2rad(60)
      joint_positions[1] = 0
      joint_positions[2] = pi/4
      joint_positions[3] = -pi/4
      joint_positions[4] = 0
      joint_positions[5] = pi/2
      joint_positions[6] = 0.2
    elif self.degrees_of_freedom == 6:
      joint_positions[0] = 0
      joint_positions[1] = 0
      joint_positions[2] = pi/2
      joint_positions[3] = pi/4
      joint_positions[4] = 0
      joint_positions[5] = pi/2
    arm_group.set_joint_value_target(joint_positions)
    
    # Plan and execute in one command
    success &= arm_group.go(wait=True)

    # Show joint positions after movement
    new_joint_positions = arm_group.get_current_joint_values()
    rospy.loginfo("Printing current joint positions after movement :")
    for p in new_joint_positions: rospy.loginfo(p)
    return success

  def get_cartesian_pose(self):
    arm_group = self.arm_group

    # Get the current pose and display it
    pose = arm_group.get_current_pose()
    rospy.loginfo("Actual cartesian pose is : ")
    rospy.loginfo(pose.pose)

    return pose.pose

  def reach_cartesian_pose(self, pose, tolerance, constraints):
    arm_group = self.arm_group
    
    # Set the tolerance
    arm_group.set_goal_position_tolerance(tolerance)

    # Set the trajectory constraint if one is specified
    if constraints is not None:
      arm_group.set_path_constraints(constraints)

    # Get the current Cartesian Position
    arm_group.set_pose_target(pose)

    # Plan and execute
    rospy.loginfo("Planning and going to the Cartesian Pose")
    return arm_group.go(wait=True)

  def reach_gripper_position(self, relative_position):
    gripper_group = self.gripper_group
    
    # We only have to move this joint because all others are mimic!
    gripper_joint = self.robot.get_joint(self.gripper_joint_name)
    gripper_max_absolute_pos = gripper_joint.max_bound()
    gripper_min_absolute_pos = gripper_joint.min_bound()
    try:
      val = gripper_joint.move(relative_position * (gripper_max_absolute_pos - gripper_min_absolute_pos) + gripper_min_absolute_pos, True)
      return val
    except:
      return False 

moveit = ExampleMoveItTrajectories()
success = moveit.is_init_success
try:
    rospy.delete_param("/kortex_examples_test_results/moveit_general_python")
except:
    pass

# args = argparse.ArgumentParser()
# args.add_argument('--camid', type=int, default=4, help='Camera ID for video capture')
# args.add_argument('--record', action='store_true', help='Record images')
# args = args.parse_args()

camid = 5
emo = Emotions(device='cuda' if torch.cuda.is_available() else 'cpu', camid=camid)
action_rec = ActionsPerf(device='cuda' if torch.cuda.is_available() else 'cpu', camid=camid)

# tc = TrajectoryControl(home=[0.3, -0.3, 0.505])

poses = MediaPipe3DPose(debug=True, translate=True)
dmp = DMPDG(n_dmps=3, n_bfs=500, T=1.5, dt=0.01, tau=1.0, tau_y=1.0, pattern="discrete", dmp_type="vanilla")
is_Approached = False
is_Extended = False
i = 0
    
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
        
#     if is_Extended and is_Approached:
#         print("Dresss")
#         break
    
# cv2.waitKey(5000)

arm_poses = []
for i in range(50):
    arm_pos, image = poses.get_arm_points()
    arm_poses.append(arm_pos)
    cv2.imshow("pose", image)
    
    cv2.waitKey(1)

arm_poses = np.array(arm_poses)
arm_poses = arm_poses[25:]#.mean(axis=0)

arm_pos= arm_poses[(arm_poses.sum(axis=-1) != 0.).sum(axis=-1) >= 1].mean(axis=0)

wrist = arm_pos[0]
elbow = arm_pos[1]

v_n = elbow - wrist
v_n /= np.linalg.norm(v_n)
ext_wrist = -v_n * 0.25 + wrist

init_traj = np.vstack((ext_wrist, arm_pos))

# y_des = make_arm_trajectory(arm_poses[25:].mean(axis=0))

y_des = make_arm_trajectory(arm_pos)
dmp.imitate_trajectory(y_des)
traj, _, _ = dmp.rollout()
traj = traj[::10].copy()
traj = np.vstack((ext_wrist, traj))
# plt.plot(traj[:, 0], traj[:, 1])
## clip values to avoid going out of workspace
traj = traj.clip(min=[0.25, -0.5, 0.1], max=[0.9, 0.5, 0.7]).copy()
plt.plot(traj[:, 0], traj[:, 1])
plt.text(traj[0,0], traj[0,1], "Start")
plt.text(traj[-1,0], traj[-1,1], "Goal")
plt.xlim(0, 1)
plt.ylim(-0.6, 0.6)
plt.show()


# success = tc.example_cartesian_waypoint_action(traj)
for tp in init_traj:
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
        
    print(tp)
    # moveit.reach_cartesian_pose
    if success:
        rospy.loginfo("Reaching Cartesian Pose...")
    
    actual_pose = moveit.get_cartesian_pose()
    for tp in traj:
        if success:
            actual_pose.position.x = tp[0]
            actual_pose.position.y = tp[1]
            actual_pose.position.z = tp[2]
            actual_pose.orientation.x = 0
            actual_pose.orientation.y = 1
            actual_pose.orientation.z = 0
            actual_pose.orientation.w = 0
            
            print(actual_pose)
            success &= moveit.reach_cartesian_pose(pose=actual_pose, tolerance=0.01, constraints=None)
            print(success)
    