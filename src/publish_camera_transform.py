#!/usr/bin/env python3

import rospy
import numpy as np
import tf2_ros
import tf
from geometry_msgs.msg import TransformStamped
import tf_conversions
from scipy.spatial.transform import Rotation as R


class CameraTransformPublisher:
    def __init__(self):
        rospy.init_node('camera_transform_publisher', anonymous=True)
        
        # Initialize tf broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Load the transformation matrix
        try:
            loaded_matrix = np.load('translation_matrix.npy')
            # The saved matrix is marker->camera, but we want camera->marker (for external_camera_frame->base_link)
            # So we take the inverse
            self.transform_matrix = loaded_matrix.copy()
            r = R.from_euler('xyz', [135., -90, -45.], degrees=True).as_matrix()
            T_color_to_depth = np.eye(4)
            T_color_to_depth[0:3, 0:3] = r
            # print(T_color_to_depth)
            self.transform_matrix = self.transform_matrix @ T_color_to_depth
            # print(self.transform_matrix)
            # self.transform_matrix[2, 3] -= 0.2  # Adjust the translation along the z-axis
            # self.transform_matrix[1, 3] -= 0.15  # Adjust the translation along the y-axis
            # self.transform_matrix[0, 3] -= 0.05  # Adjust the translation along the x-axis
            
            # print(self.transform_matrix)
            # self.transform_matrix = np.linalg.inv(loaded_matrix)
            rospy.loginfo("Successfully loaded translation_matrix.npy")
            rospy.loginfo(f"Loaded matrix shape: {loaded_matrix.shape}")
            rospy.loginfo(f"Original matrix (marker->camera):\n{loaded_matrix}")
            rospy.loginfo(f"Inverted matrix (camera->marker):\n{self.transform_matrix}")
        except FileNotFoundError:
            rospy.logerr("translation_matrix.npy not found! Please make sure the file exists.")
            rospy.signal_shutdown("Matrix file not found")
            return
        except Exception as e:
            rospy.logerr(f"Error loading translation matrix: {e}")
            rospy.signal_shutdown("Error loading matrix")
            return
        
        # Validate matrix
        if self.transform_matrix.shape != (4, 4):
            rospy.logerr(f"Invalid matrix shape: {self.transform_matrix.shape}. Expected (4, 4)")
            rospy.signal_shutdown("Invalid matrix")
            return
        # Extract rotation and translation from the 4x4 matrix
        self.rotation_matrix = self.transform_matrix[:3, :3]
        self.translation_vector = self.transform_matrix[:3, 3]
        
        rospy.loginfo(f"Translation: {self.translation_vector}")
        rospy.loginfo(f"Rotation matrix:\n{self.rotation_matrix}")
        
        # Convert rotation matrix to quaternion
        self.quaternion = tf_conversions.transformations.quaternion_from_matrix(self.transform_matrix)
        rospy.loginfo(f"Quaternion (x, y, z, w): {self.quaternion}")
        
        # Set up the transform message
        self.setup_transform()
        
        # Publishing rate (10 Hz)
        self.rate = rospy.Rate(10)
        
        rospy.loginfo("Camera transform publisher initialized successfully")
    
    def setup_transform(self):
        """Setup the static transform message"""
        self.transform_msg = TransformStamped()
        
        # Header
        self.transform_msg.header.frame_id = "base_link"  # Parent frame
        self.transform_msg.child_frame_id = "external_camera_link"  # Child frame
        
        # Note: This assumes your ArUco marker was placed at the base_link origin
        # If not, you'll need additional transformation from marker to base_link
        
        # Translation
        self.transform_msg.transform.translation.x = self.translation_vector[0]
        self.transform_msg.transform.translation.y = self.translation_vector[1]
        self.transform_msg.transform.translation.z = self.translation_vector[2]
        
        # Rotation (quaternion)
        self.transform_msg.transform.rotation.x = self.quaternion[0]
        self.transform_msg.transform.rotation.y = self.quaternion[1]
        self.transform_msg.transform.rotation.z = self.quaternion[2]
        self.transform_msg.transform.rotation.w = self.quaternion[3]
    
    def publish_transform(self):
        """Continuously publish the transform"""
        while not rospy.is_shutdown():
            # Update timestamp
            self.transform_msg.header.stamp = rospy.Time.now()
            
            # Publish the transform
            self.tf_broadcaster.sendTransform(self.transform_msg)
            
            self.rate.sleep()
    
    def print_transform_info(self):
        """Print detailed transform information"""
        rospy.loginfo("=" * 50)
        rospy.loginfo("TRANSFORM INFORMATION")
        rospy.loginfo("=" * 50)
        rospy.loginfo(f"Parent frame: {self.transform_msg.header.frame_id}")
        rospy.loginfo(f"Child frame: {self.transform_msg.child_frame_id}")
        rospy.loginfo(f"Translation (x, y, z): ({self.translation_vector[0]:.4f}, {self.translation_vector[1]:.4f}, {self.translation_vector[2]:.4f})")
        rospy.loginfo(f"Rotation (x, y, z, w): ({self.quaternion[0]:.4f}, {self.quaternion[1]:.4f}, {self.quaternion[2]:.4f}, {self.quaternion[3]:.4f})")
        
        # Convert quaternion back to Euler angles for easier understanding
        euler = tf_conversions.transformations.euler_from_quaternion(self.quaternion)
        rospy.loginfo(f"Rotation (roll, pitch, yaw) in degrees: ({np.degrees(euler[0]):.2f}, {np.degrees(euler[1]):.2f}, {np.degrees(euler[2]):.2f})")
        rospy.loginfo("=" * 50)

def main():
    try:
        # Create the publisher
        publisher = CameraTransformPublisher()
        
        # Print transform information
        publisher.print_transform_info()
        
        rospy.loginfo("Starting to publish camera transform...")
        rospy.loginfo("Use 'rosrun tf tf_echo base_link external_camera_frame' to view the transform")
        rospy.loginfo("Use 'rosrun tf view_frames' to generate frames.pdf")
        rospy.loginfo("Press Ctrl+C to stop")
        
        # Start publishing
        publisher.publish_transform()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Camera transform publisher stopped")
    except Exception as e:
        rospy.logerr(f"Error in main: {e}")

if __name__ == '__main__':
    main()
