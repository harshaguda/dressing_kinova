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

# Define available ArUco dictionaries
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


class MediaPipe3DPose:
    def __init__(self, debug=False, translate=False):
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
        # Create a context object. This object owns the handles to all connected realsense devices
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Enable streams
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.colorizer = rs.colorizer()

        # Start streaming
        profile = self.pipeline.start(config)
        # There values are needed to calculate the mapping
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.depth_min = 0.11 #meter
        self.depth_max = 1.0 #meter

        self.depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        self.depth_to_color_extrin =  profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to( profile.get_stream(rs.stream.color))
        self.color_to_depth_extrin =  profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to( profile.get_stream(rs.stream.depth))

                        

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
    def get_3d_point_from_pixel(self, idx, depth_frame, color_frame, dx, dy, x, y, translate=False):
    
        """
        Convert a 2D pixel coordinate (x,y) to a 3D point using the RealSense camera
        
        Args:
            x (int): x-coordinate of the pixel in the image
            y (int): y-coordinate of the pixel in the image
            
        Returns:
            tuple: (x, y, z) coordinates in meters in camera coordinate system
                or None if the depth value at the pixel is invalid
        """
        # x, y, dx, dy = 85, 369, 186, 314
        # 53.3 -31.6  31.6
        if not depth_frame or not color_frame:
            return None
        
        # Get depth value at the pixel (in meters)
        try:
            depth_value = depth_frame.get_distance(dx, dy)
        except RuntimeError as e:
            depth_value = 0
            print(f"Error getting depth value at pixel ({x}, {y}): {e}")
        if depth_value <= 0:
            print(f"Invalid depth at pixel ({x}, {y})")
        # Convert pixel to 3D point in camera coordinate system
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        intrinsics = rs.intrinsics()
        point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth_value)
        if translate:
            point_3d = self.t_camera_to_aruco(point_3d)
        self.previous_valid_pose[idx] = point_3d
        # print(depth_value, x, y, dx, dy)
        return point_3d


    def get_arm_points(self):

        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        frames = self.pipeline.wait_for_frames()
        # Create alignment primitive with color as its target stream:
        # It is a computationally expensive operation, maybe in future just project 
        # the required pixel to the depth frame and get x,y,z values.
        # align = rs.align(rs.stream.color)
        # frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame: return None, None

        # image = np.array(color_frame.get_data())
        image = np.asanyarray(color_frame.get_data())

        depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = self.pose.process(image)
        shoulder_verts = []
        required_landmarks = [
            # mp_pose.PoseLandmark.LEFT_SHOULDER,
            # mp_pose.PoseLandmark.LEFT_ELBOW,
            # self.mp_pose.PoseLandmark.LEFT_WRIST,
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
            dx, dy = rs.rs2_project_color_pixel_to_depth_pixel(
            depth_frame.get_data(),
            self.depth_scale,
            self.depth_min,
            self.depth_max,
            self.depth_intrin,
            self.color_intrin,
            self.depth_to_color_extrin,
            self.color_to_depth_extrin,
            list([float(x_px), float(y_px)])
        )
            shoulder_3d.append(self.get_3d_point_from_pixel(i, depth_frame, color_frame, int(dx), int(dy), x_px, y_px, self.translate))
            # print(self.get_3d_point_from_pixel(i, depth_frame, color_frame, int(dx), int(dy), x_px, y_px, self.translate)*100)
        if self.debug:
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = cv2.circle(image, center=(320, 240), radius=5, color=(255, 0, 0), thickness=-1)
            # wrist = shoulder_verts[0]
            # elbow = shoulder_verts[1]
            
            # v_n = np.array(elbow) - np.array(wrist)
            # v_n = v_n / np.linalg.norm(v_n)
            # ext_wrist = -v_n * 50 + np.array(wrist)
            # ext_wrist = ext_wrist.astype(int)
            # image = cv2.circle(image, center=(ext_wrist[0], ext_wrist[1]), radius=5, color=(0, 255, 255), thickness=-1)
            
            for i, vert in enumerate(shoulder_verts):
                x_px, y_px = vert
                # Draw a circle at the landmark position
                image = cv2.circle(image, center=(x_px, y_px), radius=5, color=(0, 0, 255), thickness=-1)
                depth_colormap = cv2.circle(depth_colormap, center=(int(dx), int(dy)), radius=5, color=(0, 255, 0), thickness=-1)
                if i != 0:
                    x_px1, y_px1 = shoulder_verts[i-1]
                    image = cv2.line(image, (x_px, y_px), (x_px1, y_px1), (0, 255, 0), 2)
            #     # put x, y, z position on the image
            #     if shoulder_3d[i] is not None:
                
            #         position_text = f"({shoulder_3d[i][0]:.2f}, {shoulder_3d[i][1]:.2f}, {shoulder_3d[i][2]:.2f})"
            #         cv2.putText(image, position_text, (x_px, y_px), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

            # print(results.pose_landmarks)
            # exit()
            # self.mp_drawing.draw_landmarks(
            #         image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            cv2.imshow('RealSense Pose Detector', image)
            cv2.imshow('RealSense Depth', depth_colormap)

            if cv2.waitKey(1) & 0xFF == 27:
                exit()
        if len(shoulder_3d) == 0:
            # return np.array(self.previous_valid_pose), image
            return np.zeros((3,3)), image
        return np.array(shoulder_3d), image

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
        T_new[:-1, 3] = [0.672 , 0.0, 0.434]  # Translation vector
        point = np.array([point[0], point[1], point[2], 1])
        point = self.translation_matrix @ point
        # print("After translation", point)
        # point = T_new @ point
        # point = point[:3] / point[3]
        return point[:-1]


if __name__ == "__main__":
    poses = MediaPipe3DPose(debug=True, translate=True)
    
    while True:
        points, _ = poses.get_arm_points()
        print(points[0]*100)
