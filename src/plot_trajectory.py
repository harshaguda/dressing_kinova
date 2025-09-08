from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
import matplotlib.pyplot as plt
import sys
import numpy as np
if len(sys.argv) >= 2:
    file_name = sys.argv[1]
else:
    file_name = "trajectory.txt"
arr = np.genfromtxt(file_name, delimiter=",")

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# ax.plot(arr[:, 0], arr[:, 1], arr[:, 2])
# ax.text(arr[0,0], arr[0,1], arr[0,2], "Start")
# ax.text(arr[-1,0], arr[-1,1], arr[-1,2], "Goal")
plt.plot(arr[:, 0], arr[:, 1])
plt.text(arr[0,0], arr[0,1], "Start")
plt.text(arr[-1,0], arr[-1,1], "Goal")
plt.show()