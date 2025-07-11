import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d

arm_pos = np.loadtxt('trajectory.txt', delimiter=',')
fig = plt.figure()
ax = Axes3D(fig)
ax.plot(arm_pos[:, 0], arm_pos[:, 1], arm_pos[:, 2])
plt.show()
