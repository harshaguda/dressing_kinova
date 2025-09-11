#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from sensor_msgs.msg import CameraInfo
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped

class RealSenseROSInterface:
    def __init__(self, camera_name="external_camera"):
        """
        Initialize RealSense ROS interface
        
        Args:
            camera_name: Name of the camera (default: "external_camera")
        """
        rospy.init_node('realsense_ros_interface', anonymous=True)
        
        self.camera_name = camera_name
        self.bridge = CvBridge()
        
        # Latest images
        self.color_image = None
        self.depth_image = None
        self.color_info = None
        self.depth_info = None
        
        # TF2 buffer for transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Subscribe to camera topics
        self.setup_subscribers()
        
        rospy.loginfo(f"RealSense ROS interface initialized for camera: {camera_name}")
        
    def setup_subscribers(self):
        """Setup ROS subscribers for color and depth images"""
        
        # Topic names
        color_topic = f"/{self.camera_name}/color/image_raw"
        depth_topic = f"/{self.camera_name}/aligned_depth_to_color/image_raw"
        color_info_topic = f"/{self.camera_name}/color/camera_info"
        depth_info_topic = f"/{self.camera_name}/aligned_depth_to_color/camera_info"
        
        rospy.loginfo(f"Subscribing to:")
        rospy.loginfo(f"  Color: {color_topic}")
        rospy.loginfo(f"  Depth: {depth_topic}")
        rospy.loginfo(f"  Color info: {color_info_topic}")
        rospy.loginfo(f"  Depth info: {depth_info_topic}")
        
        # Synchronized subscribers for color and depth
        color_sub = message_filters.Subscriber(color_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        
        # Synchronize color and depth images
        ts = message_filters.TimeSynchronizer([color_sub, depth_sub], 10)
        ts.registerCallback(self.image_callback)
        
        # Camera info subscribers
        rospy.Subscriber(color_info_topic, CameraInfo, self.color_info_callback)
        rospy.Subscriber(depth_info_topic, CameraInfo, self.depth_info_callback)
        
    def image_callback(self, color_msg, depth_msg):
        """Callback for synchronized color and depth images"""
        try:
            # Convert ROS images to OpenCV format
            self.color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            
            # Store timestamps
            self.color_timestamp = color_msg.header.stamp
            self.depth_timestamp = depth_msg.header.stamp
            self.color_frame_id = color_msg.header.frame_id
            self.depth_frame_id = depth_msg.header.frame_id
            
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
    
    def color_info_callback(self, msg):
        """Callback for color camera info"""
        self.color_info = msg
        
    def depth_info_callback(self, msg):
        """Callback for depth camera info"""
        self.depth_info = msg
    
    def get_images(self):
        """
        Get the latest color and depth images
        
        Returns:
            tuple: (color_image, depth_image) or (None, None) if no images available
        """
        return self.color_image, self.depth_image
    
    def get_camera_info(self):
        """
        Get camera calibration info
        
        Returns:
            tuple: (color_info, depth_info)
        """
        return self.color_info, self.depth_info
    
    def pixel_to_3d_point(self, u, v, depth_value=None):
        """
        Convert pixel coordinates to 3D point in camera frame
        
        Args:
            u, v: Pixel coordinates
            depth_value: Depth value in meters (if None, will get from depth image)
            
        Returns:
            numpy.array: [x, y, z] coordinates in camera frame or None if invalid
        """
        if self.depth_image is None or self.depth_info is None:
            rospy.logwarn("No depth image or camera info available")
            return None
            
        # Get depth value if not provided
        if depth_value is None:
            if u < 0 or v < 0 or u >= self.depth_image.shape[1] or v >= self.depth_image.shape[0]:
                return None
            depth_value = self.depth_image[v, u] / 1000.0  # Convert mm to meters
        
        if depth_value <= 0:
            return None
            
        # Camera intrinsics
        fx = self.depth_info.K[0]
        fy = self.depth_info.K[4]
        cx = self.depth_info.K[2]
        cy = self.depth_info.K[5]
        
        # Convert to 3D
        x = (u - cx) * depth_value / fx
        y = (v - cy) * depth_value / fy
        z = depth_value
        
        return np.array([x, y, z])
    
    def transform_point_to_base_link(self, point_3d, source_frame=None):
        """
        Transform a 3D point from camera frame to base_link frame
        
        Args:
            point_3d: [x, y, z] coordinates in camera frame
            source_frame: Source frame name (if None, uses color frame_id)
            
        Returns:
            numpy.array: [x, y, z] coordinates in base_link frame or None if transform fails
        """
        if source_frame is None:
            source_frame = self.color_frame_id
            
        try:
            # Create point stamped message
            point_stamped = PointStamped()
            point_stamped.header.frame_id = source_frame
            point_stamped.header.stamp = rospy.Time.now()
            point_stamped.point.x = point_3d[0]
            point_stamped.point.y = point_3d[1]
            point_stamped.point.z = point_3d[2]
            
            # Transform to base_link
            transformed_point = self.tf_buffer.transform(point_stamped, "base_link", timeout=rospy.Duration(1.0))
            
            return np.array([
                transformed_point.point.x,
                transformed_point.point.y,
                transformed_point.point.z
            ])
            
        except Exception as e:
            rospy.logwarn(f"Transform failed: {e}")
            return None
    
    def display_images(self, window_name="RealSense"):
        """
        Display color and depth images in OpenCV windows
        
        Args:
            window_name: Base name for the windows
        """
        if self.color_image is not None:
            cv2.imshow(f"{window_name} - Color", self.color_image)
            
        if self.depth_image is not None:
            # Normalize depth for display
            depth_display = cv2.convertScaleAbs(self.depth_image, alpha=0.03)
            depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
            cv2.imshow(f"{window_name} - Depth", depth_colormap)
    
    def save_images(self, prefix="realsense"):
        """
        Save current color and depth images
        
        Args:
            prefix: Filename prefix
        """
        if self.color_image is not None:
            color_filename = f"{prefix}_color.png"
            cv2.imwrite(color_filename, self.color_image)
            rospy.loginfo(f"Saved color image: {color_filename}")
            
        if self.depth_image is not None:
            depth_filename = f"{prefix}_depth.png"
            cv2.imwrite(depth_filename, self.depth_image)
            rospy.loginfo(f"Saved depth image: {depth_filename}")

def main():
    try:
        # Initialize the RealSense interface
        rs_interface = RealSenseROSInterface("external_camera")
        
        rospy.loginfo("Waiting for images...")
        
        # Wait for first images
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            color_img, depth_img = rs_interface.get_images()
            
            if color_img is not None and depth_img is not None:
                rospy.loginfo("Received images!")
                rospy.loginfo(f"Color image shape: {color_img.shape}")
                rospy.loginfo(f"Depth image shape: {depth_img.shape}")
                
                # Display images
                rs_interface.display_images()
                
                # Example: Get 3D point at center of image
                h, w = color_img.shape[:2]
                center_u, center_v = w // 2, h // 2
                point_3d = rs_interface.pixel_to_3d_point(center_u, center_v)
                
                if point_3d is not None:
                    rospy.loginfo(f"Center pixel ({center_u}, {center_v}) -> 3D point: {point_3d}")
                    
                    # Transform to base_link
                    point_base_link = rs_interface.transform_point_to_base_link(point_3d)
                    if point_base_link is not None:
                        rospy.loginfo(f"Point in base_link frame: {point_base_link}")
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    rs_interface.save_images()
                    
            rate.sleep()
            
    except rospy.ROSInterruptException:
        rospy.loginfo("RealSense ROS interface stopped")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
