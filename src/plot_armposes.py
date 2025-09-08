import numpy as np
import matplotlib.pyplot as plt
from pose_estimation import MediaPipe3DPose
from mpl_toolkits.mplot3d import Axes3D
poses = MediaPipe3DPose(debug=True, translate=True)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
while True:
    points, _ = poses.get_arm_points()
    wrist = points[0]
    elbow = points[1]

    v_n = elbow - wrist
    v_n /= np.linalg.norm(v_n)
    ext_wrist = (-v_n * 0.25 + wrist)
    plt.cla()
    # print(points[0]*100)
    points = np.vstack((ext_wrist, points))
    print(points[0], ext_wrist)
    ax.plot3D(points[:,0]*100, points[:,1]*100, points[:,2]*100, 'o-')
    ax.scatter3D(ext_wrist[0]*100, ext_wrist[1]*100, ext_wrist[2]*100, c="r")
    ax.set_xlim([0, 90])
    ax.set_ylim([-50, 100])
    ax.set_zlim([0, 100])
    plt.draw()
    plt.pause(0.1)