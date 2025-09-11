import rospy
from simple_realsense_ros import SimpleRealSenseROS
import cv2

if __name__ == "__main__":
    try:
        # Initialize the Simple RealSense ROS interface
        rs_interface = SimpleRealSenseROS("external_camera")
        
        rospy.loginfo("Waiting for images...")
        
        # Wait for first images
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            color_img, depth_img = rs_interface.get_images()
            
            if color_img is not None and depth_img is not None:
                # rospy.loginfo("Received images!")
                # rospy.loginfo(f"Color image shape: {color_img.shape}")
                # rospy.loginfo(f"Depth image shape: {depth_img.shape}")
                
                # Display images
                cv2.imshow("Color Image", color_img)
                
                # Normalize depth for display
                depth_display = cv2.convertScaleAbs(depth_img, alpha=0.03)
                depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                cv2.imshow("Depth Image", depth_colormap)
                
                # cv2.waitKey(1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            rate.sleep()
            
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()