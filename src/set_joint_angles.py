#!/usr/bin/env python3

import rospy
import math
from kortex_driver.srv import PlayJointTrajectory, PlayJointTrajectoryRequest
from kortex_driver.msg import JointAngle, JointAngles, ConstrainedJointAngles

def set_joint_angles(joint_angles_deg):
    """
    Set joint angles for Kinova arm using ros_kortex (without MoveIt)
    
    Args:
        joint_angles_deg: List of 7 joint angles in degrees
    """
    rospy.init_node('joint_angle_setter')
    
    # Wait for service
    service_name = '/my_gen3/base/play_joint_trajectory'
    rospy.loginfo(f"Waiting for service {service_name}...")
    rospy.wait_for_service(service_name)
    play_joint_trajectory = rospy.ServiceProxy(service_name, PlayJointTrajectory)
    
    # Create joint angles
    joint_angles = JointAngles()
    
    # Add each joint angle
    for i, angle_deg in enumerate(joint_angles_deg):
        joint_angle = JointAngle()
        joint_angle.joint_identifier = i
        joint_angle.value = angle_deg  # Kinova uses degrees
        joint_angles.joint_angles.append(joint_angle)
    
    # Create constrained joint angles
    constrained_joint_angles = ConstrainedJointAngles()
    constrained_joint_angles.joint_angles = joint_angles
    
    # Create and send request
    req = PlayJointTrajectoryRequest()
    req.input = constrained_joint_angles
    
    try:
        rospy.loginfo(f"Sending joint angles: {joint_angles_deg}")
        result = play_joint_trajectory(req)
        rospy.loginfo("Joint trajectory sent successfully")
        return True
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to send joint trajectory: {e}")
        return False

def main():
    # Example joint angles in degrees [joint0, joint1, joint2, joint3, joint4, joint5, joint6]
    # These angles should put the arm in a safe position
    desired_angles = [45, 15, 180, -130, 0, 55, 90]  # Replace with your desired angles
    
    try:
        success = set_joint_angles(desired_angles)
        if success:
            rospy.loginfo("Joint angles set successfully!")
        else:
            rospy.logerr("Failed to set joint angles")
    except rospy.ROSInterruptException:
        rospy.loginfo("Program interrupted")

if __name__ == "__main__":
    main()
