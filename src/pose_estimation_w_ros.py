from calendar import c
import enum
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import pyrealsense2 as rs
import cv2
import time
import math
import numpy as np
import rospy

import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose, PoseArray, PointStamped


from simple_realsense_ros import SimpleRealSenseROS
from projection_rs import ImageListener

class MediaPipe3DPose:
    def __init__(self, debug=False, translate=False):
        
        self.pose_publisher = rospy.Publisher('/pose_arm', PoseArray, queue_size=10)
                    
    
        # self.real_sense = SimpleRealSenseROS("external_camera")
        self.real_sense = ImageListener(
            depth_image_topic="/external_camera/aligned_depth_to_color/image_raw",
            depth_info_topic="/external_camera/aligned_depth_to_color/camera_info",
            color_image_topic="/external_camera/color/image_raw",
            color_info_topic="/external_camera/color/camera_info"
        )
        
        self.debug = debug
        self.translate = translate
        self.no_smooth_landmarks = False
        self.static_image_mode = True
        self.model_complexity = 1
        self.min_detection_confidence = 0.5
        self.min_tracking_confidence = 0.5
        self.model_path = 'pose_landmarker.task'

        self.mp_drawing = mp.solutions.drawing_utils

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            smooth_landmarks=self.no_smooth_landmarks,
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
            )
        
        # Get RealSense equivalent values from ROS
        self.depth_scale = 0.001  # 0.001 (1mm = 0.001m)
        self.depth_min = 0.11      # 0.11 meter
        self.depth_max = 1.0       # 1.0 meter

        # Wait for camera info to be available
        rospy.loginfo("Waiting for camera info...")
        
        # Get intrinsics from ROS camera info
        self.color_intrin = self.real_sense.color_intrinsics
        # self.depth_intrin = self.real_sense.intrinsics

        # For ROS, we don't need extrinsics if using aligned depth
        self.depth_to_color_extrin = None  # Not needed for aligned depth
        self.color_to_depth_extrin = None  # Not needed for aligned depth

        self.previous_valid_pose = [[0,0,0], [0,0,0], [0,0,0]]
        self.aruco_x, self.aruco_y = None, None
        
        # self._init_translation_matrix()
        self.translation_matrix  = np.load("/home/userlab/iri_lab/iri_ws/src/dressing_kinova/src/translation_matrix.npy")
        
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]
            # print(pose_landmarks, idx)
            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_image


    # def get_3d_point_from_pixel(self, idx, depth_frame, color_frame, dx, dy, x, y, translate=False):
    def get_3d_point_from_pixel(self, idx, x, y, translate=False):
        """
        Convert a 2D pixel coordinate (x,y) to a 3D point using the RealSense camera
        
        Args:
            x (int): x-coordinate of the pixel in the image
            y (int): y-coordinate of the pixel in the image
            
        Returns:
            tuple: (x, y, z) coordinates in meters in camera coordinate system
                or None if the depth value at the pixel is invalid
        """
        # if not depth_frame or not color_frame:
        #     return None
        
        # Get depth value at the pixel using ROS interface
        try:
            depth_value, du, dv = self.real_sense.get_depth_at_color_pixel(x, y)
        except RuntimeError as e:
            depth_value = 0
            print(f"Error getting depth value at pixel ({x}, {y}): {e}")
        
        if depth_value is None or depth_value <= 0:
            print(f"Invalid depth at pixel ({x}, {y})")
            return np.array([0, 0, 0]), None, None
        
        # Convert pixel to 3D point using ROS camera intrinsics
        point_3d = self.real_sense.get_3d_point_from_color_pixel(x, y)
        
        if point_3d is None:
            return self.previous_valid_pose[idx]
            
        if translate:
            point_3d = self.t_camera_to_aruco(point_3d)
        
        self.previous_valid_pose[idx] = point_3d
        return point_3d, du, dv


    def get_arm_points(self):
        # Get images from ROS instead of direct pipeline
        color_image, depth_image = self.real_sense.get_images()
        if color_image is None or depth_image is None:
            return None, None

        # Use the ROS images directly
        image = color_image.copy()
        depth_colormap = depth_image.copy()

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = self.pose.process(image)
        shoulder_verts = []
        required_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            # self.mp_pose.PoseLandmark.RIGHT_WRIST,
            # self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            # self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
        ]
        if results.pose_landmarks is not None:
            for landmark in required_landmarks:
                landmark_px = results.pose_landmarks.landmark[landmark]
                x_px = min(math.floor(landmark_px.x * w), w - 1)
                y_px = min(math.floor(landmark_px.y * h), h - 1)
                shoulder_verts.append([x_px, y_px])
        shoulder_3d = []
        for i, vert in enumerate(shoulder_verts):
            x_px, y_px = vert
            point_3d, du, dv = self.get_3d_point_from_pixel(i, x_px, y_px, self.translate)
            shoulder_3d.append(point_3d)
            
        if self.debug:
            image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            for i, vert in enumerate(shoulder_verts):
                x_px, y_px = vert
                # Draw a circle at the landmark position
                image = cv2.circle(image, center=(x_px, y_px), radius=5, color=(0, 0, 255), thickness=-1)
                # Create depth colormap for visualization
                depth_display = cv2.convertScaleAbs(depth_image, alpha=0.03)
                depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                if du is not None and dv is not None:
                    depth_colormap = cv2.circle(depth_colormap, center=(int(du), int(dv)), radius=5, color=(0, 255, 0), thickness=-1)
                if i != 0:
                    x_px1, y_px1 = shoulder_verts[i-1]
                    image = cv2.line(image, (x_px, y_px), (x_px1, y_px1), (0, 255, 0), 2)
                    
            cv2.imshow('RealSense Pose Detector', image)
            cv2.imshow('RealSense Depth', depth_colormap)

            if cv2.waitKey(1) & 0xFF == 27:
                exit()
        if len(shoulder_3d) == 0:
            return np.zeros((3,3)), image
        
        self.publish_poses(shoulder_3d)
        return np.array(shoulder_3d), image

    def publish_poses(self, traj):
        pa = PoseArray()
        for tp in traj:
            p = Pose()
            if np.sum(tp) == 0:
                continue
            p.position.x = tp[0]
            p.position.y = tp[1]
            p.position.z = tp[2]
            pa.poses.append(p)
            
        pa.header.frame_id = "base_link"
        self.pose_publisher.publish(pa)    
    
    def t_camera_to_aruco(self, point):
        """
        Transform a point from camera coordinates to ArUco marker coordinates.
        
        Args:
        point: 3D point in camera coordinates
        Trans: Transformation matrix from camera to ArUco marker
        
        Returns:
        point: Transformed 3D point in ArUco marker coordinates
        """
        T_new = np.eye(4)
        T_new[:-1, 3] = [0.2 , 0.15, 0.05]  # Translation vector
        # self.transform_matrix[2, 3] -= 0.2  # Adjust the translation along the z-axis
        # self.transform_matrix[1, 3] -= 0.15  # Adjust the translation along the y-axis
        # self.transform_matrix[0, 3] -= 0.05  # Adjust the translation along the x-axis
        point = np.array([point[0], point[1], point[2], 1])
        # point = (self.translation_matrix @ T_new) @ point
        point = self.translation_matrix @ point
        # print(self.translation_matrix, T_new, self.translation_matrix @ T_new)
        # print("After translation", point)
        # point = T_new @ point
        # point = point[:3] / point[3]
        return point[:-1]


if __name__ == "__main__":
    rospy.init_node('pose_estimation_node', anonymous=True)
    
    poses = MediaPipe3DPose(debug=True, translate=True)
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        points, _ = poses.get_arm_points()
        # print(points)
        rate.sleep()