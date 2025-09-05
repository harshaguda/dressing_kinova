import rospy
from full_trajectory import TrajectoryControl
import numpy as np
import sys
from delta_pose_control import DeltaPoseControl
from pose_control import ExampleCartesianActionsWithNotifications

pc = ExampleCartesianActionsWithNotifications()
pc.main()
# dpc = DeltaPoseControl(home=[0.3, -0.3, 0.505])
tc = TrajectoryControl(home=[0.3, -0.3, 0.505])
if len(sys.argv) >= 2:
    traj_path = sys.argv[1]
else:
    traj_path = "test_trajectory.txt"
traj = np.genfromtxt(traj_path, delimiter=",")
print(tc.is_init_success)
success = tc.example_cartesian_waypoint_action(traj)
# # For testing purposes
rospy.set_param("/kortex_examples_test_results/waypoint_action_python", success)

# for t in traj:
#     # dpc.set_cartesian_pose(t[0], t[1], t[2])
#     pc.set_pose(t[0], t[1], t[2])
#     rospy.sleep(0.1)

# For testing purposes
# rospy.set_param("/kortex_examples_test_results/cartesian_poses_with_notifications_python", True)

