# dressing_kinova
## Start Kinova control
```
roslaunch kortex_driver kortex_driver.launch gripper:=robotiq_2f_85
```
For simulation,
```
roslaunch kortex_gazebo spawn_kortex_robot.launch gripper:=robotiq_2f_85
```
Download the model from [here](https://drive.google.com/drive/folders/1AP9-GGl6UEebppQfkFCq9kCkgNn64Gxd?usp=drive_link)

## Setup origin with ArUco marker

```
python originAruco.py --dict DICT_5X5_50 --use_realsense
```
