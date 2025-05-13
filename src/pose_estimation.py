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

class MediaPipe3DPose:
    def __init__(self, debug=False):
        self.debug = debug
        self.no_smooth_landmarks = False
        self.static_image_mode = True
        self.model_complexity = 1
        self.min_detection_confidence = 0.5
        self.min_tracking_confidence = 0.5
        self.model_path = 'pose_landmarker.task'


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
        self.pipeline.start(config)

        self.previous_valid_pose = [[0,0,0], [0,0,0], [0,0,0]]

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

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



    def get_3d_point_from_pixel(self, idx, depth_frame, color_frame, x, y):
        """
        Convert a 2D pixel coordinate (x,y) to a 3D point using the RealSense camera
        
        Args:
            x (int): x-coordinate of the pixel in the image
            y (int): y-coordinate of the pixel in the image
            
        Returns:
            tuple: (x, y, z) coordinates in meters in camera coordinate system
                or None if the depth value at the pixel is invalid
        """
        if not depth_frame or not color_frame:
            return None
        
        # Get depth value at the pixel (in meters)
        try:
            depth_value = depth_frame.get_distance(x, y)
        except RuntimeError as e:
            print(f"Error getting depth value at pixel ({x}, {y}): {e}")
            return self.previous_valid_pose[idx]
        if depth_value <= 0:
            print(f"Invalid depth at pixel ({x}, {y})")
            return self.previous_valid_pose[idx]
        
        # Convert pixel to 3D point in camera coordinate system
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth_value)
        
        self.previous_valid_pose[idx] = point_3d
        return point_3d


    def get_arm_points(self):

        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        frames = self.pipeline.wait_for_frames()
        # Create alignment primitive with color as its target stream:
        # It is a computationally expensive operation, maybe in future just project 
        # the required pixel to the depth frame and get x,y,z values.
        align = rs.align(rs.stream.color)
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame: return None

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
        image_rows, image_cols, _ = image.shape
        shoulder_verts = []
        required_landmarks = [
            # mp_pose.PoseLandmark.LEFT_SHOULDER,
            # mp_pose.PoseLandmark.LEFT_ELBOW,
            # mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_WRIST
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
            shoulder_3d.append(self.get_3d_point_from_pixel(i, depth_frame, color_frame, x_px, y_px))
        
        if self.debug:
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # mp_drawing.draw_landmarks(
            #     image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            for i, vert in enumerate(shoulder_verts):
                x_px, y_px = vert
                # Draw a circle at the landmark position
                image = cv2.circle(image, center=(x_px, y_px), radius=5, color=(0, 255, 0), thickness=-1)
                depth_colormap = cv2.circle(depth_colormap, center=(x_px, y_px), radius=5, color=(0, 255, 0), thickness=-1)
                # put x, y, z position on the image
                if shoulder_3d[i] is not None:
                
                    position_text = f"({shoulder_3d[i][0]:.2f}, {shoulder_3d[i][1]:.2f}, {shoulder_3d[i][2]:.2f})"
                    cv2.putText(image, position_text, (x_px, y_px), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

            
            cv2.imshow('RealSense Pose Detector', image)
            cv2.imshow('RealSense Depth', depth_colormap)

            if cv2.waitKey(1) & 0xFF == 27:
                exit()
        return shoulder_3d
    
if __name__ == "__main__":
    poses = MediaPipe3DPose(debug=True)
    
    while True:
        points = poses.get_arm_points()
        print(points)