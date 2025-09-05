import cv2
import numpy as np
from pose_estimation import MediaPipe3DPose
from actions_perf import ActionsPerf
from emotions import Emotions
import torch

cam_id = 4
emo = Emotions(device='cuda' if torch.cuda.is_available() else 'cpu', camid=cam_id)
action_rec = ActionsPerf(device='cuda' if torch.cuda.is_available() else 'cpu', camid=cam_id)
# poses = MediaPipe3DPose(debug=True, translate=True)
to_dress = False
is_Approached = False
is_Extended = False
while True:
    # arm_pos, image = poses.get_arm_points()
    e_image, emotion, engagement = emo.predict_emotions()
    action_label, image = action_rec.predict_actions(e_image)

    print(is_Extended, is_Approached, action_label)
    if (not is_Approached) and (not is_Extended) and (action_label == "Approach"):
        is_Approached = True
    if (is_Approached) and (action_label == "ExtendArm"):
        is_Extended = True
    
    if is_Extended and is_Approached:
        print("Dresss")
        break
    # if (action_label == "ExtendArm") & (emotion in ["Neutral", "Happiness"]):
    #     print("Dressing action detected")
    # else:
    #     print("No dressing action")
    # Display the frame with detected faces and emotions
    if e_image is not None:
        # new_image = np.concatenate((image, e_image), axis=1)
        cv2.imshow('Emotion Detection', image)
        # cv2.imwrite(f"img2/image_{i}.png", new_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break