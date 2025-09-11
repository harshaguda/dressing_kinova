import rospy
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import sys
import os
import numpy as np
import pyrealsense2 as rs2
import cv2
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped

if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2

class ImageListener:
    def __init__(self, depth_image_topic, depth_info_topic, color_image_topic, color_info_topic):
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber(depth_image_topic, msg_Image, self.imageDepthCallback)
        self.sub_info = rospy.Subscriber(depth_info_topic, CameraInfo, self.imageDepthInfoCallback)
        self.color_sub = rospy.Subscriber(color_image_topic, msg_Image, self.imageColorCallback)
        self.color_info_sub = rospy.Subscriber(color_info_topic, CameraInfo, self.imageColorInfoCallback)
        confidence_topic = depth_image_topic.replace('depth', 'confidence')
        self.sub_conf = rospy.Subscriber(confidence_topic, msg_Image, self.confidenceCallback)
        
        # Initialize attributes
        self.depth_intrinsics = None
        self.color_intrinsics = None
        self.color_image = None
        self.depth_image = None
        self.pix = None
        self.pix_grade = None
        self.matrix_coefficients = None
        self.distortion_coefficients = None
        
        # TF for extrinsics
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.depth_to_color_transform = None
        
        # Camera frame names (adjust these to match your setup)
        self.depth_frame = "external_camera_depth_optical_frame"
        self.color_frame = "external_camera_color_optical_frame"

    def imageColorCallback(self, msg):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return
    
    def imageDepthCallback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
            self.depth_image = cv_image.copy()
        except CvBridgeError as e:
            print(e)
            return
        except ValueError as e:
            return

    def confidenceCallback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            grades = np.bitwise_and(cv_image >> 4, 0x0f)
            if (self.pix):
                self.pix_grade = grades[self.pix[1], self.pix[0]]
        except CvBridgeError as e:
            print(e)
            return

    def imageDepthInfoCallback(self, cameraInfo):
        try:
            if self.depth_intrinsics:
                return
            self.depth_intrinsics = rs2.intrinsics()
            self.depth_intrinsics.width = cameraInfo.width
            self.depth_intrinsics.height = cameraInfo.height
            self.depth_intrinsics.ppx = cameraInfo.K[2]
            self.depth_intrinsics.ppy = cameraInfo.K[5]
            self.depth_intrinsics.fx = cameraInfo.K[0]
            self.depth_intrinsics.fy = cameraInfo.K[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.depth_intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.depth_intrinsics.model = rs2.distortion.kannala_brandt4
            self.depth_intrinsics.coeffs = [i for i in cameraInfo.D]
            self._get_extrinsics()
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
            self.matrix_coefficients = cameraInfo.K
            # self.color_intrinsics.distortion_coefficients = cameraInfo.D
            if cameraInfo.distortion_model == 'plumb_bob':
                self.color_intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.color_intrinsics.model = rs2.distortion.kannala_brandt4
            self.color_intrinsics.coeffs = [i for i in cameraInfo.D]
            self._get_extrinsics()
        except CvBridgeError as e:
            print(e)
            return

    def _get_extrinsics(self):
        """Get extrinsics between depth and color cameras from TF"""
        if self.depth_intrinsics is None or self.color_intrinsics is None:
            return
            
        try:
            # Get transform from depth to color frame
            transform = self.tf_buffer.lookup_transform(
                self.color_frame, self.depth_frame, rospy.Time(), rospy.Duration(1.0))
            
            # Convert to extrinsics format
            self.depth_to_color_transform = self._transform_to_extrinsics(transform)
            rospy.loginfo("Extrinsics obtained from TF")
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Could not get extrinsics from TF: {e}. Assuming aligned cameras.")
            self.depth_to_color_transform = None

    def _transform_to_extrinsics(self, transform):
        """Convert ROS transform to RealSense extrinsics format"""
        extrinsics = rs2.extrinsics()
        
        # Translation
        extrinsics.translation = [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ]
        
        # Rotation (quaternion to rotation matrix)
        q = transform.transform.rotation
        # Convert quaternion to rotation matrix
        q_norm = np.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        q.x /= q_norm
        q.y /= q_norm
        q.z /= q_norm
        q.w /= q_norm
        
        rotation_matrix = [
            1 - 2*(q.y**2 + q.z**2), 2*(q.x*q.y - q.z*q.w), 2*(q.x*q.z + q.y*q.w),
            2*(q.x*q.y + q.z*q.w), 1 - 2*(q.x**2 + q.z**2), 2*(q.y*q.z - q.x*q.w),
            2*(q.x*q.z - q.y*q.w), 2*(q.y*q.z + q.x*q.w), 1 - 2*(q.x**2 + q.y**2)
        ]
        extrinsics.rotation = rotation_matrix
        
        return extrinsics

    def project_color_pixel_to_depth_pixel(self, color_u, color_v, depth_value):
        """
        Project a color pixel to depth pixel coordinates using camera calibration
        
        Args:
            color_u: x coordinate in color image
            color_v: y coordinate in color image  
            depth_value: depth value at the corresponding 3D point
            
        Returns:
            tuple: (depth_u, depth_v) coordinates in depth image, or None if projection fails
        """
        if self.color_intrinsics is None or self.depth_intrinsics is None:
            rospy.logwarn("Camera intrinsics not available")
            return None
            
        try:
            if self.depth_to_color_transform is None:
                # Cameras are aligned, use direct mapping
                return (color_u, color_v)
            
            # Deproject color pixel to 3D point in color camera frame
            color_point_3d = rs2.rs2_deproject_pixel_to_point(
                self.color_intrinsics, [color_u, color_v], depth_value)
            
            # Transform 3D point from color frame to depth frame
            depth_point_3d = rs2.rs2_transform_point_to_point(
                self.depth_to_color_transform, color_point_3d)
            
            # Project 3D point to depth pixel
            depth_pixel = rs2.rs2_project_point_to_pixel(
                self.depth_intrinsics, depth_point_3d)
            
            # Check bounds
            if (0 <= depth_pixel[0] < self.depth_intrinsics.width and 
                0 <= depth_pixel[1] < self.depth_intrinsics.height):
                return (int(depth_pixel[0]), int(depth_pixel[1]))
            else:
                return None
                
        except Exception as e:
            rospy.logerr(f"Error in pixel projection: {e}")
            return None

    def get_depth_at_color_pixel(self, color_u, color_v):
        """
        Get depth value at a color pixel coordinate
        
        Args:
            color_u: x coordinate in color image
            color_v: y coordinate in color image
            
        Returns:
            float: depth value in meters, or None if no valid depth
        """
        if self.depth_image is None:
            rospy.logwarn("No depth image available")
            return None, None, None
            
        # Check bounds for color image
        if (color_u < 0 or color_u >= self.color_intrinsics.width or
            color_v < 0 or color_v >= self.color_intrinsics.height):
            return None, None, None
            
        if self.depth_to_color_transform is None:
            # Cameras are aligned - direct mapping
            # Scale coordinates if image sizes are different
            if (self.color_intrinsics.width != self.depth_intrinsics.width or 
                self.color_intrinsics.height != self.depth_intrinsics.height):
                
                scale_x = self.depth_intrinsics.width / self.color_intrinsics.width
                scale_y = self.depth_intrinsics.height / self.color_intrinsics.height
                depth_u = int(color_u * scale_x)
                depth_v = int(color_v * scale_y)
            else:
                depth_u, depth_v = color_u, color_v
        else:
            # Non-aligned cameras - need to estimate depth for projection
            # Use average depth in neighborhood for initial estimate
            neighborhood_size = 5
            total_depth = 0
            valid_pixels = 0
            
            for du in range(-neighborhood_size, neighborhood_size + 1):
                for dv in range(-neighborhood_size, neighborhood_size + 1):
                    test_u = color_u + du
                    test_v = color_v + dv
                    if (0 <= test_u < self.color_intrinsics.width and 
                        0 <= test_v < self.color_intrinsics.height):
                        # Rough estimate using aligned assumption for initial depth
                        scale_x = self.depth_intrinsics.width / self.color_intrinsics.width
                        scale_y = self.depth_intrinsics.height / self.color_intrinsics.height
                        est_depth_u = int(test_u * scale_x)
                        est_depth_v = int(test_v * scale_y)
                        
                        if (0 <= est_depth_u < self.depth_image.shape[1] and 
                            0 <= est_depth_v < self.depth_image.shape[0]):
                            depth_val = self.depth_image[est_depth_v, est_depth_u]
                            if depth_val > 0:
                                total_depth += depth_val
                                valid_pixels += 1
            
            if valid_pixels == 0:
                return None, None, None
                
            avg_depth = total_depth / valid_pixels
            
            # Use this average depth for projection
            depth_coords = self.project_color_pixel_to_depth_pixel(color_u, color_v, avg_depth)
            if depth_coords is None:
                return None
            depth_u, depth_v = depth_coords
        
        # Get depth value
        if (0 <= depth_u < self.depth_image.shape[1] and 
            0 <= depth_v < self.depth_image.shape[0]):
            depth_value = self.depth_image[depth_v, depth_u]
            if depth_value > 0:
                return depth_value / 1000.0, depth_u, depth_v  # Convert mm to meters
        
        return None, depth_u, depth_v

    def get_3d_point_from_color_pixel(self, color_u, color_v):
        """
        Get 3D point coordinates from color pixel
        
        Args:
            color_u: x coordinate in color image
            color_v: y coordinate in color image
            
        Returns:
            list: [x, y, z] coordinates in meters in color camera frame, or None
        """
        depth, du, dv = self.get_depth_at_color_pixel(color_u, color_v)
        if depth is None:
            return None
            
        if self.color_intrinsics is None:
            rospy.logwarn("Color intrinsics not available")
            return None
            
        try:
            # Convert depth back to mm for RealSense function
            depth_mm = depth * 1000.0
            point_3d = rs2.rs2_deproject_pixel_to_point(
                self.color_intrinsics, [color_u, color_v], depth_mm)
            # Convert back to meters
            return [point_3d[0]/1000.0, point_3d[1]/1000.0, point_3d[2]/1000.0]
        except Exception as e:
            rospy.logerr(f"Error in deprojection: {e}")
            return None

    def get_images(self):
        """
        Get the latest color and depth images
        
        Returns:
            tuple: (color_image, depth_image) or (None, None) if no images available
        """
        return self.color_image, self.depth_image
    
def main():
    depth_image_topic = '/external_camera/depth/image_rect_raw'
    depth_info_topic = '/external_camera/depth/camera_info'
    color_image_topic = '/external_camera/color/image_raw'
    color_info_topic = '/external_camera/color/camera_info'

    print ('')
    print ('Non-aligned depth projection demo')
    print ('----------------------------------')
    print ('Demonstrating depth lookup from color pixels with non-aligned cameras')
    print ('')
    
    listener = ImageListener(depth_image_topic, depth_info_topic, color_image_topic, color_info_topic)
    
    # Wait for images and intrinsics to be available
    print("Waiting for images and camera info...")
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        color, depth = listener.get_images()
        
        if (color is not None and depth is not None and 
            listener.color_intrinsics is not None and listener.depth_intrinsics is not None):
            
            print("Images and intrinsics received, starting demo...")
            
            # Test pixel coordinates in color image
            test_color_u, test_color_v = 328, 539
            
            # Get depth at color pixel
            depth_value, du, dv = listener.get_depth_at_color_pixel(test_color_u, test_color_v)
            if depth_value is not None:
                print(f"Depth at color pixel ({test_color_u}, {test_color_v}): {depth_value:.3f} meters")
                
                # Get 3D point
                point_3d = listener.get_3d_point_from_color_pixel(test_color_u, test_color_v)
                if point_3d is not None:
                    print(f"3D point: [{point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f}] meters")
            else:
                print(f"No valid depth at color pixel ({test_color_u}, {test_color_v})")
            
            # Visualize
            color_display = color.copy()
            cv2.circle(color_display, (test_color_u, test_color_v), 5, (0, 0, 255), -1)
            cv2.circle(depth, (du, dv), 4, (65535, 0, 0), 2)
            
            cv2.putText(color_display, f"Test pixel", (test_color_u+10, test_color_v-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imshow("Color Image", color_display)
            cv2.imshow("Depth Image", depth)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.signal_shutdown("User requested shutdown.")
        else:
            missing = []
            if color is None: missing.append("color")
            if depth is None: missing.append("depth") 
            if listener.color_intrinsics is None: missing.append("color_intrinsics")
            if listener.depth_intrinsics is None: missing.append("depth_intrinsics")
            print(f"Waiting for: {', '.join(missing)}")
            
        rate.sleep()
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    node_name = os.path.basename(sys.argv[0]).split('.')[0]
    rospy.init_node(node_name)
    main()