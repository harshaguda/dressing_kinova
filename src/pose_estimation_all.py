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

def draw_landmarks_on_image(rgb_image, detection_result):
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


class Args:
    def __init__(self):
        self.no_smooth_landmarks = False
        self.static_image_mode = True
        self.model_complexity = 1
        self.min_detection_confidence = 0.5
        self.min_tracking_confidence = 0.5
        self.model_path = 'pose_landmarker.task'


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
args = Args()
pose = mp_pose.Pose(
    smooth_landmarks=args.no_smooth_landmarks,
    static_image_mode=args.static_image_mode,
    model_complexity=args.model_complexity,
    min_detection_confidence=args.min_detection_confidence,
    min_tracking_confidence=args.min_tracking_confidence
    )


# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
config = rs.config()

# Enable streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
colorizer = rs.colorizer()

# Start streaming
pipeline.start(config)

class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

state = AppState()

def get_3d_point_from_pixel(depth_frame, color_frame, x, y):
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
    
    if x < 0 or y < 0 or x >= depth_frame.get_width() or y >= depth_frame.get_height():
        print(f"Pixel ({x}, {y}) is out of bounds for the depth frame.")
        return None
    # Get depth value at the pixel (in meters)
    depth_value = depth_frame.get_distance(x, y)
    if depth_value <= 0:
        print(f"Invalid depth at pixel ({x}, {y})")
        return None
    
    # Convert pixel to 3D point in camera coordinate system
    depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth_value)
    
    return point_3d

try:
    prev_frame_time = 0
    x_px, y_px = 320, 240
    debug = True
    while True:
        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        frames = pipeline.wait_for_frames()
        # Create alignment primitive with color as its target stream:
        # It is a computationally expensive operation, maybe in future just project 
        # the required pixel to the depth frame and get x,y,z values.
        align = rs.align(rs.stream.color)
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame: continue

        # image = np.array(color_frame.get_data())
        image = np.asanyarray(color_frame.get_data())
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image)
        image_rows, image_cols, _ = image.shape
        shoulder_verts = []
        required_landmarks = [
            # mp_pose.PoseLandmark.LEFT_SHOULDER,
            # mp_pose.PoseLandmark.LEFT_ELBOW,
            # mp_pose.PoseLandmark.LEFT_WRIST,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_WRIST
        ]
        if results.pose_landmarks is not None:
            for landmark in required_landmarks:
                landmark_px = results.pose_landmarks.landmark[landmark]
                x_px = min(math.floor(landmark_px.x * w), w - 1)
                y_px = min(math.floor(landmark_px.y * h), h - 1)
                shoulder_verts.append([x_px, y_px])
        shoulder_3d = []
        for vert in shoulder_verts:
            x_px, y_px = vert
            shoulder_3d.append(get_3d_point_from_pixel(depth_frame, color_frame, x_px, y_px))

        # print(f"Shoulder Vertices: {shoulder_3d}")
        if len(shoulder_3d) > 0:
            wrist3d = shoulder_3d[2]
            if wrist3d is not None:
                # print(f"Distance from camera: {np.linalg.norm(wrist3d)}")
                cam_wrist = np.linalg.norm(wrist3d)
                if cam_wrist < 0.2:
                    print("Dressing")

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        current_time = time.time()
        fps = 1 / (current_time - prev_frame_time)
        prev_frame_time = current_time
        
        cv2.putText(image, "FPS: %.0f" % fps, (7, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
        
        if debug:
            cv2.imshow('RealSense Pose Detector', image)

        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    pose.close()
    pipeline.stop()
