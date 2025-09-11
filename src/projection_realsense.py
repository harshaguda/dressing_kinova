import rospy
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import sys
import os
import numpy as np
import pyrealsense2 as rs2
import cv2

if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2

class ImageListener:
    def __init__(self, depth_image_topic, depth_info_topic, color_image_topic):
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber(depth_image_topic, msg_Image, self.imageDepthCallback)
        self.sub_info = rospy.Subscriber(depth_info_topic, CameraInfo, self.imageDepthInfoCallback)
        self.color_sub = rospy.Subscriber(color_image_topic, msg_Image, self.imageColorCallback)
        confidence_topic = depth_image_topic.replace('depth', 'confidence')
        self.sub_conf = rospy.Subscriber(confidence_topic, msg_Image, self.confidenceCallback)
        self.intrinsics = None
        self.pix = None
        self.pix_grade = None
        # Initialize image attributes
        self.color_image = None
        self.depth_image = None

    def imageColorCallback(self, msg):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # cv2.imshow("Color Image", self.color_image)
            # cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)
            return
    
    def imageDepthCallback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
            # pick one pixel among all the pixels with the closest range:
            indices = np.array(np.where(cv_image == cv_image[cv_image > 0].min()))[:,0]
            # pix = (indices[1], indices[0])
            # self.pix = pix
            # pix = (400, 300)
            
            # line = '\rDepth at pixel(%3d, %3d): %7.1f(mm).' % (pix[0], pix[1], cv_image[pix[1], pix[0]])
            # # print(pix)
            # if self.intrinsics:
            #     depth = cv_image[pix[1], pix[0]]
            #     result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [pix[0], pix[1]], depth)
            #     line += '  Coordinate: %8.2f %8.2f %8.2f.' % (result[0], result[1], result[2])
            # if (not self.pix_grade is None):
            #     line += ' Grade: %2d' % self.pix_grade
            # line += '\r'
            # cv2.circle(cv_image, pix, 4, (65535, 0, 0), 2)
            self.depth_image = cv_image.copy()
            
            # cv2.imshow("Depth Image", cv_image)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     rospy.signal_shutdown("User requested shutdown.")
            # sys.stdout.write(line)
            # sys.stdout.flush()

        except CvBridgeError as e:
            print(e)
            return
        except ValueError as e:
            return
        
    def deproject3d_pixel_to_point(self, u, v):
        if self.intrinsics is None:
            rospy.logwarn("Intrinsics not available yet.")
            return None
        try:
            depth = self.depth_image[u, v]
            point_3d = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], depth)
            return point_3d
        except Exception as e:
            rospy.logerr(f"Error in deprojection: {e}")
            return None

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
            # print(self.intrinsics)
        except CvBridgeError as e:
            print(e)
            return

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

    print ('')
    print ('show_center_depth.py')
    print ('--------------------')
    print ('App to demontrate the usage of the /camera/depth topics.')
    print ('')
    print ('Application subscribes to %s and %s topics.' % (depth_image_topic, depth_info_topic))
    print ('Application then calculates and print the range to the closest object.')
    print ('If intrinsics data is available, it also prints the 3D location of the object')
    print ('If a confedence map is also available in the topic %s, it also prints the confidence grade.' % depth_image_topic.replace('depth', 'confidence'))
    print ('')
    
    listener = ImageListener(depth_image_topic, depth_info_topic, color_image_topic)
    
    # Wait for images to be available
    print("Waiting for images...")
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        color, depth = listener.get_images()
        
        if color is not None and depth is not None:
            print("Images received, displaying...")
            # 328, 539
            xr, yr = depth.shape[0]/color.shape[0], depth.shape[1]/color.shape[1]
            cv2.circle(color, (328, 539), 4, (0, 0, 255), -1)
            # print(int(328*xr), int(539*yr))
            print(depth.shape)
            cv2.circle(depth, (int(328*xr), int(539*yr)), 4, (65535, 0, 0), 2)
            # cv2.circle(depth, (400,300), 4, (65535, 0, 0), -1)
            cv2.imshow("Color Image", color)
            cv2.imshow("Depth Image", depth)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.signal_shutdown("User requested shutdown.")
        else:
            print("Waiting for images... (color: {}, depth: {})".format(
                color is not None, depth is not None))
            
        rate.sleep()
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    node_name = os.path.basename(sys.argv[0]).split('.')[0]
    rospy.init_node(node_name)
    main()
