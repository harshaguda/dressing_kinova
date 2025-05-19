#!/usr/bin/env python3
###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2019 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import rospy
import time
from kortex_driver.srv import *
from kortex_driver.msg import *
import tf

class DeltaPoseControl:
    def __init__(self):
        try:
            rospy.init_node('example_cartesian_poses_with_notifications_python')

            self.tf_listener = tf.TransformListener()

            self.HOME_ACTION_IDENTIFIER = 2

            self.action_topic_sub = None
            self.all_notifs_succeeded = True

            self.all_notifs_succeeded = True

            # Get node params
            self.robot_name = rospy.get_param('~robot_name', "my_gen3")

            rospy.loginfo("Using robot_name " + self.robot_name)

            # Init the action topic subscriber
            self.action_topic_sub = rospy.Subscriber("/" + self.robot_name + "/action_topic", ActionNotification, self.cb_action_topic)
            self.last_action_notif_type = None

            # Init the services
            clear_faults_full_name = '/' + self.robot_name + '/base/clear_faults'
            rospy.wait_for_service(clear_faults_full_name)
            self.clear_faults = rospy.ServiceProxy(clear_faults_full_name, Base_ClearFaults)

            read_action_full_name = '/' + self.robot_name + '/base/read_action'
            rospy.wait_for_service(read_action_full_name)
            self.read_action = rospy.ServiceProxy(read_action_full_name, ReadAction)

            execute_action_full_name = '/' + self.robot_name + '/base/execute_action'
            rospy.wait_for_service(execute_action_full_name)
            self.execute_action = rospy.ServiceProxy(execute_action_full_name, ExecuteAction)

            set_cartesian_reference_frame_full_name = '/' + self.robot_name + '/control_config/set_cartesian_reference_frame'
            rospy.wait_for_service(set_cartesian_reference_frame_full_name)
            self.set_cartesian_reference_frame = rospy.ServiceProxy(set_cartesian_reference_frame_full_name, SetCartesianReferenceFrame)

            activate_publishing_of_action_notification_full_name = '/' + self.robot_name + '/base/activate_publishing_of_action_topic'
            rospy.wait_for_service(activate_publishing_of_action_notification_full_name)
            self.activate_publishing_of_action_notification = rospy.ServiceProxy(activate_publishing_of_action_notification_full_name, OnNotificationActionTopic)

            self.req_handle = 1001
            self._init_controller()
        except:
            self.is_init_success = False
        else:
            self.is_init_success = True

    def cb_action_topic(self, notif):
        self.last_action_notif_type = notif.action_event

    def wait_for_action_end_or_abort(self):
        while not rospy.is_shutdown():
            if (self.last_action_notif_type == ActionEvent.ACTION_END):
                rospy.loginfo("Received ACTION_END notification")
                return True
            elif (self.last_action_notif_type == ActionEvent.ACTION_ABORT):
                rospy.loginfo("Received ACTION_ABORT notification")
                self.all_notifs_succeeded = False
                return False
            else:
                time.sleep(0.01)

    def example_clear_faults(self):
        try:
            self.clear_faults()
        except rospy.ServiceException:
            rospy.logerr("Failed to call ClearFaults")
            return False
        else:
            rospy.loginfo("Cleared the faults successfully")
            rospy.sleep(2.5)
            return True

    def get_ee_pose(self):
        # Get the end effector pose
        try:
            trans, rot = self.tf_listener.lookupTransform('/base_link', '/end_effector_link', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Failed to get the end effector pose")
            return False
        else:
            return trans, rot
        
        rospy.sleep(0.25)

    def home_the_robot(self):
        # The Home Action is used to home the robot. It cannot be deleted and is always ID #2:
        req = ReadActionRequest()
        req.input.identifier = self.HOME_ACTION_IDENTIFIER
        self.last_action_notif_type = None
        try:
            res = self.read_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ReadAction")
            return False
        # Execute the HOME action if we could read it
        else:
            # What we just read is the input of the ExecuteAction service
            req = ExecuteActionRequest()
            req.input = res.output
            rospy.loginfo("Sending the robot home...")
            try:
                self.execute_action(req)
            except rospy.ServiceException:
                rospy.logerr("Failed to call ExecuteAction")
                return False
            else:
                return self.wait_for_action_end_or_abort()

    def ex_set_cartesian_reference_frame(self):
        # Prepare the request with the frame we want to set
        req = SetCartesianReferenceFrameRequest()
        req.input.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED

        # Call the service
        try:
            self.set_cartesian_reference_frame()
        except rospy.ServiceException:
            rospy.logerr("Failed to call SetCartesianReferenceFrame")
            return False
        else:
            rospy.loginfo("Set the cartesian reference frame successfully")
            return True

        # Wait a bit
        rospy.sleep(0.25)

    def subscribe_to_a_robot_notification(self):
        # Activate the publishing of the ActionNotification
        req = OnNotificationActionTopicRequest()
        rospy.loginfo("Activating the action notifications...")
        try:
            self.activate_publishing_of_action_notification(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call OnNotificationActionTopic")
            return False
        else:
            rospy.loginfo("Successfully activated the Action Notifications!")

        rospy.sleep(1.0)

        return True

    def _init_linear_pose(self):
        # Prepare and send pose 1
        self.my_cartesian_speed = CartesianSpeed()
        self.my_cartesian_speed.translation = 0.1 # m/s
        self.my_cartesian_speed.orientation = 15  # deg/s

        self.my_constrained_pose = ConstrainedPose()
        self.my_constrained_pose.constraint.oneof_type.speed.append(self.my_cartesian_speed)

        self.my_constrained_pose.target_pose.x = 0.374
        self.my_constrained_pose.target_pose.y = 0.311
        self.my_constrained_pose.target_pose.z = 0.305
        self.my_constrained_pose.target_pose.theta_x = 0
        self.my_constrained_pose.target_pose.theta_y = 180
        self.my_constrained_pose.target_pose.theta_z = 0

        self.req = ExecuteActionRequest()
        self.req.input.oneof_action_parameters.reach_pose.append(self.my_constrained_pose)
        self.req.input.name = "pose1"
        self.req.input.handle.action_type = ActionType.REACH_POSE
        self.req.input.handle.identifier = self.req_handle
        self.req_handle += 1

        rospy.loginfo(f"Sending pose {self.req_handle - 1}...")
        self.last_action_notif_type = None
        try:
            self.execute_action(self.req)
        except rospy.ServiceException:
            rospy.logerr("Failed to send pose 1")
            success = False
        else:
            rospy.loginfo("Waiting for pose 1 to finish...")
        self.wait_for_action_end_or_abort()

    def _init_controller(self):
        #*******************************************************************************
        # Make sure to clear the robot's faults else it won't move if it's already in fault
        # print("clear faults", self.clear_faults())
        self.example_clear_faults()
        #*******************************************************************************

        #*******************************************************************************
        # Start the example from the Home position
        # self.home_the_robot()
        #*******************************************************************************

        #*******************************************************************************
        # Set the reference frame to "Mixed"
        self.ex_set_cartesian_reference_frame()

        #*******************************************************************************
        # Subscribe to ActionNotification's from the robot to know when a cartesian pose is finished
        self.subscribe_to_a_robot_notification()

        #*******************************************************************************

        self._init_linear_pose()    
    
    def set_cartesian_pose(self, x, y, z):
        # cartesian speed
        # Prepare and send pose 2
        self.req.input.handle.identifier = 1002
        self.req.input.name = f"{self.req_handle}"
        self.my_constrained_pose.target_pose.x += x
        self.my_constrained_pose.target_pose.y += y
        self.my_constrained_pose.target_pose.z += z

        self.req.input.oneof_action_parameters.reach_pose[0] = self.my_constrained_pose

        rospy.loginfo(f"Sending pose {self.req_handle}...")
        self.req_handle += 1
        self.last_action_notif_type = None
        try:
            self.execute_action(self.req)
        except rospy.ServiceException:
            rospy.logerr("Failed to send pose 2")
            success = False
        else:
            rospy.loginfo(f"Waiting for pose {self.req_handle} to finish...")

        self.wait_for_action_end_or_abort()

    def main(self):
        # For testing purposes
        success = self.is_init_success
        try:
            rospy.delete_param("/kortex_examples_test_results/cartesian_poses_with_notifications_python")
        except:
            pass

        if success:

            #*******************************************************************************
            # Make sure to clear the robot's faults else it won't move if it's already in fault
            # print("clear faults", self.clear_faults())
            success &= self.example_clear_faults()
            #*******************************************************************************
            
            #*******************************************************************************
            # Start the example from the Home position
            # success &= self.home_the_robot()
            #*******************************************************************************

            #*******************************************************************************
            # Set the reference frame to "Mixed"
            success &= self.ex_set_cartesian_reference_frame()

            #*******************************************************************************
            # Subscribe to ActionNotification's from the robot to know when a cartesian pose is finished
            success &= self.subscribe_to_a_robot_notification()

            #*******************************************************************************
            
            self._init_linear_pose()
            # self.set_cartesian_pose(0.1, 0, 0)
            # self.set_cartesian_pose(0, 0.1, 0) 
            # self.set_cartesian_pose(0, 0, 0.1)

            success &= self.all_notifs_succeeded

            success &= self.all_notifs_succeeded

        # For testing purposes
        rospy.set_param("/kortex_examples_test_results/cartesian_poses_with_notifications_python", success)

        if not success:
            rospy.logerr("The example encountered an error.")

if __name__ == "__main__":
    ex = DeltaPoseControl()
    ex.main()
