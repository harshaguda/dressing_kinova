#!/usr/bin/env python3
import rospy
import math
from kortex_driver.srv import ExecuteAction, ExecuteActionRequest
from kortex_driver.msg import Action, Reach, ConstrainedPose, JointAngles, JointAngle

def set_joint_angles_with_action(joint_angles_deg):
    rospy.init_node('joint_angle_action_setter')
    
    # Wait for service
    rospy.wait_for_service('/my_gen3/base/execute_action')
    execute_action = rospy.ServiceProxy('/my_gen3/base/execute_action', ExecuteAction)
    
    # Create action
    action = Action()
    action.name = "Joint Angle Movement"
    action.application_data = ""
    
    # Create reach action
    reach = Reach()
    reach.constraint.oneof_type.joint_angles.joint_angles = []
    
    for i, angle_deg in enumerate(joint_angles_deg):
        joint_angle = JointAngle()
        joint_angle.joint_identifier = i
        joint_angle.value = angle_deg  # Kinova uses degrees
        reach.constraint.oneof_type.joint_angles.joint_angles.append(joint_angle)
    
    action.oneof_action_parameters.reach.append(reach)
    
    # Send request
    req = ExecuteActionRequest()
    req.input = action
    
    try:
        result = execute_action(req)
        rospy.loginfo("Action executed successfully")
        return True
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to execute action: {e}")
        return False

# Example usage:
if __name__ == "__main__":
    # Set joint angles in degrees [joint0, joint1, joint2, joint3, joint4, joint5, joint6]
    desired_angles = [0, 30, 45, 0, 0, 0, 0]  # Replace with your desired angles
    set_joint_angles_with_action(desired_angles)