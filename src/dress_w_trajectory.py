import cv2
import numpy as np
import rospy
from pose_estimation import MediaPipe3DPose
from dmp_kinova import DMPDG, make_arm_trajectory
from full_trajectory import TrajectoryControl
from emotions import Emotions
from actions_perf import ActionsPerf
import torch

emo = Emotions(device='cuda' if torch.cuda.is_available() else 'cpu', camid=10)
action_rec = ActionsPerf(device='cuda' if torch.cuda.is_available() else 'cpu', camid=10)

tc = TrajectoryControl(home=[0.3, -0.3, 0.505])

poses = MediaPipe3DPose(debug=True, translate=True)
dmp = DMPDG(n_dmps=3, n_bfs=500, T=1.5, dt=0.01, tau=1.0, tau_y=1.0, pattern="discrete", dmp_type="vanilla")

for i in range(50):
    arm_pos, image = poses.get_arm_points()
    e_image, emotion, engagement = emo.predict_emotions()
    action_label, image_a = action_rec.predict_actions(e_image)
        
    # Display the frame with detected faces and emotions
    if e_image is not None:
        new_image = np.concatenate((image, image_a), axis=1)
        cv2.imshow('Emotion Detection', new_image)
        cv2.imwrite(f"img2/image_{i}.png", new_image)
    cv2.waitKey(1)

if (action_label == "ExtendArm") & (emotion in ["Neutral", "Happiness"]):
    print("Dressing action detected")
    print("Doing motion generation")
    y_des = make_arm_trajectory(arm_pos)
    dmp.imitate_trajectory(y_des)
    traj, _, _ = dmp.rollout()
    traj = traj[::10].copy()
    traj = traj.clip(-0.6, 0.55)

    np.savetxt("trajectory.txt", traj, delimiter=',')
    success = tc.example_cartesian_waypoint_action(traj)
    # For testing purposes
    rospy.set_param("/kortex_examples_test_results/waypoint_action_python", success)
else:
    print("No dressing action")
    print(f"Detected action: {action_label}, emotion: {emotion}")
    print("No motion generation")
    cv2.waitKey(1)
    cv2.destroyAllWindows()
