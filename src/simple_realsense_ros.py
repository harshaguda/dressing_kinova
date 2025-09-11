#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
import pyrealsense2 as rs2

class SimpleRealSenseROS:
    def __init__(self, camera_name="external_camera", intrinsics=None, color_intrinsics=None):
        """
        Simple RealSense ROS interface to get color and depth images
        
        Args:
            camera_name: Name of the camera (default: "external_camera")
        """
        rospy.init_node('simple_realsense_ros', anonymous=True)
        
        self.intrinsics = intrinsics
        self.color_intrinsics = color_intrinsics
        self.camera_name = camera_name
        self.bridge = CvBridge()
        
        # Latest images
        self.color_image = None
        self.depth_image = None
        
        # Camera info
        self.color_info = None
        self.depth_info = None
        
        # TF2 buffer for extrinsics
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # RealSense equivalent values
        self.depth_scale = 0.001  # Default: 1mm = 0.001m
        self.depth_min = 0.11  # meter
        self.depth_max = 1.0   # meter
        
        # Extrinsics (will be computed from TF)
        self.depth_to_color_extrin = None
        self.color_to_depth_extrin = None
        
        # Setup subscribers
        color_topic = f"/{camera_name}/color/image_raw"
        depth_topic = f"/{camera_name}/depth/image_rect_raw"
        color_info_topic = f"/{camera_name}/color/camera_info"
        depth_info_topic = f"/{camera_name}/depth/camera_info"
        
        rospy.loginfo(f"Subscribing to:")
        rospy.loginfo(f"  Color: {color_topic}")
        rospy.loginfo(f"  Depth: {depth_topic}")
        rospy.loginfo(f"  Color info: {color_info_topic}")
        rospy.loginfo(f"  Depth info: {depth_info_topic}")
        
        self.color_sub = rospy.Subscriber(color_topic, Image, self.color_callback)
        self.depth_sub = rospy.Subscriber(depth_topic, Image, self.depth_callback)
        self.color_info_sub = rospy.Subscriber(color_info_topic, CameraInfo, self.imageColorInfoCallback)
        self.depth_info_sub = rospy.Subscriber(depth_info_topic, CameraInfo, self.imageDepthInfoCallback)
        
        rospy.loginfo("Simple RealSense ROS interface initialized")
        
    def color_callback(self, msg):
        """Callback for color image"""
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Color CvBridge Error: {e}")
    
    def depth_callback(self, msg):
        """Callback for depth image"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except CvBridgeError as e:
            rospy.logerr(f"Depth CvBridge Error: {e}")
    
    def color_info_callback(self, msg):
        """Callback for color camera info"""
        self.color_info = msg
        self._update_extrinsics()
        
    def depth_info_callback(self, msg):
        """Callback for depth camera info"""
        self.depth_info = msg
        self._update_extrinsics()
    
    def _update_extrinsics(self):
        """Update extrinsics from TF tree"""
        if self.color_info is None or self.depth_info is None:
            return
            
        try:
            # Get transform from depth to color frame
            color_frame = f"{self.camera_name}_color_optical_frame"
            depth_frame = f"{self.camera_name}_depth_optical_frame"
            
            # Try to get transform
            transform = self.tf_buffer.lookup_transform(
                color_frame, depth_frame, rospy.Time(0), rospy.Duration(1.0)
            )
            
            # Convert to RealSense-like extrinsics format
            self.depth_to_color_extrin = self._tf_to_realsense_extrinsics(transform)
            
            # Get inverse transform
            transform_inv = self.tf_buffer.lookup_transform(
                depth_frame, color_frame, rospy.Time(0), rospy.Duration(1.0)
            )
            self.color_to_depth_extrin = self._tf_to_realsense_extrinsics(transform_inv)
            
            # rospy.loginfo("Successfully extracted extrinsics from TF")
            
        except Exception as e:
            rospy.logwarn(f"Could not get extrinsics from TF: {e}")
            # Use identity transform as fallback for aligned cameras
            self.depth_to_color_extrin = {
                'rotation': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                'translation': [0.0, 0.0, 0.0]
            }
            self.color_to_depth_extrin = {
                'rotation': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                'translation': [0.0, 0.0, 0.0]
            }
    
    def _tf_to_realsense_extrinsics(self, transform):
        """Convert TF transform to RealSense extrinsics format"""
        # Extract rotation quaternion
        q = transform.transform.rotation
        quat = [q.x, q.y, q.z, q.w]
        
        # Convert quaternion to rotation matrix
        from tf_conversions import transformations
        rot_matrix = transformations.quaternion_matrix(quat)[:3, :3]
        
        # Extract translation
        t = transform.transform.translation
        translation = [t.x, t.y, t.z]
        
        return {
            'rotation': rot_matrix.flatten().tolist(),
            'translation': translation
        }
    
    def get_images(self):
        """
        Get the latest color and depth images
        
        Returns:
            tuple: (color_image, depth_image) or (None, None) if no images available
        """
        return self.color_image, self.depth_image
    
    def get_depth_at_pixel(self, color_u, color_v):
        """
        Get depth value at a color pixel coordinate
        
        Args:
            color_u, color_v: Pixel coordinates in color image
            
        Returns:
            float: Depth value in meters, or None if unavailable
        """
        if self.depth_image is None:
            return None
            
        # Convert color pixel to depth pixel coordinates
        depth_coords = self.project_color_pixel_to_depth_pixel(color_u, color_v)
        if depth_coords is None:
            return None
            
        depth_u, depth_v = depth_coords
        
        # Check bounds
        if (depth_u < 0 or depth_u >= self.depth_image.shape[1] or
            depth_v < 0 or depth_v >= self.depth_image.shape[0]):
            return None
            
        # Get depth value (convert from mm to meters)
        depth_mm = self.depth_image[depth_v, depth_u]
        if depth_mm == 0:  # Invalid depth
            return None
            
        return depth_mm / 1000.0  # Convert to meters
    
    def get_intrinsics_as_realsense_format(self):
        """
        Get camera intrinsics in RealSense-like format
        
        Returns:
            dict: Dictionary with color_intrin and depth_intrin like RealSense
        """
        intrinsics = {}
        
        if self.color_info:
            intrinsics['color_intrin'] = {
                'fx': self.color_info.K[0],
                'fy': self.color_info.K[4], 
                'ppx': self.color_info.K[2],
                'ppy': self.color_info.K[5],
                'width': self.color_info.width,
                'height': self.color_info.height,
                'coeffs': list(self.color_info.D)
            }
            
        if self.depth_info:
            intrinsics['depth_intrin'] = {
                'fx': self.depth_info.K[0],
                'fy': self.depth_info.K[4],
                'ppx': self.depth_info.K[2], 
                'ppy': self.depth_info.K[5],
                'width': self.depth_info.width,
                'height': self.depth_info.height,
                'coeffs': list(self.depth_info.D)
            }
            
        return intrinsics
    
    def imageDepthInfoCallback(self, cameraInfo):
        try:
            if self.intrinsics:
                return
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = cameraInfo.width
            self.intrinsics.height = cameraInfo.height
            self.intrinsics.ppx = cameraInfo.K[2]
            self.intrinsics.ppy = cameraInfo.K[5]
            self.intrinsics.fx = cameraInfo.K[0]
            self.intrinsics.fy = cameraInfo.K[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in cameraInfo.D]
            print(self.intrinsics)
        except CvBridgeError as e:
            print(e)
            return
        
    def imageColorInfoCallback(self, cameraInfo):
        try:
            if self.color_intrinsics:
                return
            self.color_intrinsics = rs2.intrinsics()
            self.color_intrinsics.width = cameraInfo.width
            self.color_intrinsics.height = cameraInfo.height
            self.color_intrinsics.ppx = cameraInfo.K[2]
            self.color_intrinsics.ppy = cameraInfo.K[5]
            self.color_intrinsics.fx = cameraInfo.K[0]
            self.color_intrinsics.fy = cameraInfo.K[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.color_intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.color_intrinsics.model = rs2.distortion.kannala_brandt4
            self.color_intrinsics.coeffs = [i for i in cameraInfo.D]
            print(self.color_intrinsics)
        except CvBridgeError as e:
            print(e)
            return
    def project_color_pixel_to_depth_pixel(self, color_u, color_v):
        """
        Project color pixel to depth pixel using extrinsics
        
        Args:
            color_u, color_v: Color pixel coordinates
            
        Returns:
            tuple: (depth_u, depth_v) or None if projection fails
        """
        if (self.color_info is None or self.depth_info is None or 
            self.color_to_depth_extrin is None):
            # Fallback: assume aligned cameras
            return color_u, color_v
            
        try:
            # This is a simplified version - for full accuracy you'd need depth value
            # For now, assume a default depth or use median depth
            default_depth = 0.5  # meters
            
            # Deproject color pixel to 3D point in color frame
            color_fx = self.color_info.K[0]
            color_fy = self.color_info.K[4]
            color_cx = self.color_info.K[2]
            color_cy = self.color_info.K[5]
            
            # 3D point in color camera frame
            x_color = (color_u - color_cx) * default_depth / color_fx
            y_color = (color_v - color_cy) * default_depth / color_fy
            z_color = default_depth
            
            # Transform to depth camera frame using extrinsics
            rot = np.array(self.color_to_depth_extrin['rotation']).reshape(3, 3)
            trans = np.array(self.color_to_depth_extrin['translation'])
            
            point_color = np.array([x_color, y_color, z_color])
            point_depth = rot @ point_color + trans
            
            # Project to depth pixel
            depth_fx = self.depth_info.K[0]
            depth_fy = self.depth_info.K[4]
            depth_cx = self.depth_info.K[2]
            depth_cy = self.depth_info.K[5]
            
            depth_u = int(point_depth[0] * depth_fx / point_depth[2] + depth_cx)
            depth_v = int(point_depth[1] * depth_fy / point_depth[2] + depth_cy)
            
            # Clamp to image bounds
            depth_u = max(0, min(depth_u, self.depth_info.width - 1))
            depth_v = max(0, min(depth_v, self.depth_info.height - 1))
            
            return depth_u, depth_v
            
        except Exception as e:
            rospy.logwarn(f"Color to depth projection failed: {e}")
            return color_u, color_v
    
    def deproject_pixel_to_point(self, u, v, depth_value):
        """
        Convert pixel coordinates and depth to 3D point
        
        Args:
            u, v: Pixel coordinates
            depth_value: Depth value in meters
            
        Returns:
            list: [x, y, z] coordinates in camera frame
        """
        return rs2.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], depth_value)

    
    def display_images(self):
        """Display color and depth images"""
        if self.color_image is not None:
            cv2.imshow("Color Image", self.color_image)
            
        if self.depth_image is not None:
            # Normalize depth for display
            depth_display = cv2.convertScaleAbs(self.depth_image, alpha=0.03)
            depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
            cv2.imshow("Depth Image", depth_colormap)

def main():
    try:
        # Initialize the interface
        rs = SimpleRealSenseROS("external_camera")
        
        rospy.loginfo("Waiting for images... Press 'q' to quit, 's' to save")
        
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            color_img, depth_img = rs.get_images()
            
            if color_img is not None and depth_img is not None:
                # Display images
                rs.display_images()
                
                # Example: Get depth at center
                h, w = color_img.shape[:2]
                center_u, center_v = w // 2, h // 2
                depth_value = rs.get_depth_at_pixel(center_u, center_v)
                
                if depth_value is not None:
                    # Draw center point and depth value
                    cv2.circle(color_img, (center_u, center_v), 5, (0, 255, 0), -1)
                    cv2.putText(color_img, f"Depth: {depth_value:.3f}m", 
                               (center_u + 10, center_v), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                
                cv2.imshow("Color with Depth Info", color_img)
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save images
                    cv2.imwrite("color_image.png", color_img)
                    cv2.imwrite("depth_image.png", depth_img)
                    rospy.loginfo("Images saved!")
                    
            rate.sleep()
            
    except rospy.ROSInterruptException:
        rospy.loginfo("Simple RealSense ROS interface stopped")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
