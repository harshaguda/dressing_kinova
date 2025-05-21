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
        self.aruco_x, self.aruco_y = None, None
        self.get_realsense_intrinsics()
        self.translation_matrix = None
        
        # self._init_translation_matrix()
        self.translation_matrix  = np.load("/home/userlab/iri_lab/iri_ws/src/dressing_kinova/src/translation_matrix.npy")
        # self.translation_matrix = np.array([[8.957507014274597168e-01,-1.949338763952255249e-01,3.995389938354492188e-01,-9.345052391290664673e-02],
        #                            [-4.422885775566101074e-01,-3.001050353050231934e-01,8.451732397079467773e-01,-7.846307754516601562e-01],
        #                            [-4.484923928976058960e-02,-9.337760806083679199e-01,-3.550363183021545410e-01,6.802514195442199707e-01],
        #                            [0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])
    def _init_translation_matrix(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not depth_frame or not color_frame:
            print("No frames")
        frame = np.asanyarray(color_frame.get_data())
        depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
        rvec, vert = None, None
        flag = True
        while flag:
            output, rvec, vert = self.detect_aruco_markers(frame, 
                                                    "DICT_5X5_50",  # replace with an args
                                                    depth_frame=depth_frame, 
                                                    color_frame=color_frame,
                                                    depth_colormap=depth_colormap)
            if (rvec is None) or (np.sum(vert) == 0):
                flag = True
            else:
                flag = False
            # cv2.imshow("Original Image Feed", output)
            # cv2.waitKey(1)
        self.translation_matrix = self.get_translation_matrix(rvec, vert)
    
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



    def get_3d_point_from_pixel(self, idx, depth_frame, color_frame, x, y, translate=False):
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
        if translate:
            point_3d = self.t_camera_to_aruco(point_3d)
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
        image_rows, image_cols, _ = image.shape
        shoulder_verts = []
        required_landmarks = [
            # mp_pose.PoseLandmark.LEFT_SHOULDER,
            # mp_pose.PoseLandmark.LEFT_ELBOW,
            # mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
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
            shoulder_3d.append(self.get_3d_point_from_pixel(i, depth_frame, color_frame, x_px, y_px, self.translate))
        
        if self.debug:
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # mp_drawing.draw_landmarks(
            #     image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow("Original Image Feed", image)
            image = cv2.circle(image, center=(320, 240), radius=5, color=(255, 0, 0), thickness=-1)
            if self.aruco_x is not None:
                image = cv2.circle(image, center=(self.aruco_x, self.aruco_y), radius=5, color=(255, 0, 0), thickness=-1)
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
        if len(shoulder_3d) == 0:
            return np.array(self.previous_valid_pose), image
        return np.array(shoulder_3d), image
    
    def detect_aruco_markers(
            self,
            frame, 
            aruco_dict_type,
            depth_frame=None,
            color_frame=None,
            depth_colormap=None,
            ):
        """
        Detect ArUco markers in the image
        
        Args:
            frame: Input image frame
            aruco_dict_type: Type of ArUco dictionary
            matrix_coefficients: Camera matrix (optional, for pose estimation)
            distortion_coefficients: Distortion coefficients (optional, for pose estimation)
            
        Returns:
            frame: Output image with detected markers
        """
        print("detecting aruco marker")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get ArUco dictionary
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_dict_type])
        parameters = cv2.aruco.DetectorParameters()
        
        # Detect ArUco markers
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
        rvec, tvec, vert = None, None, None
        # Draw detected markers if any
        if ids is not None and len(ids) > 0:
            # rvec, tvec, _ = cv2.aruco.(corners, 0.05, matrix_coefficients, distortion_coefficients)
            x_px, y_px = int(corners[0].mean(axis=1)[0, 0]), int(corners[0].mean(axis=1)[0, 1])
            self.aruco_x, self.aruco_y = x_px, y_px
            if self.translation_matrix is None:
                translate = False
            else:
                translate = True
            vert = self.get_3d_point_from_pixel(0, depth_frame, color_frame, x_px, y_px, translate)
            self.aruco_vert = vert
            # exit()
            # If camera calibration is provided, estimate pose
            if self.matrix_coefficients is not None and self.distortion_coefficients is not None:
                # Define marker size (in meters)
                marker_size = 0.05
                
                # For each detected marker
                for i in range(len(ids)):
                    # Define marker corners in 3D space (marker coordinate system)
                    objPoints = np.array([
                        [-marker_size/2, marker_size/2, 0],
                        [marker_size/2, marker_size/2, 0],
                        [marker_size/2, -marker_size/2, 0],
                        [-marker_size/2, -marker_size/2, 0]
                    ], dtype=np.float32)
                    
                    # Get 2D corners from detected marker
                    imgPoints = corners[i][0].astype(np.float32)
                    
                    # Solve for pose
                    success, rvec, tvec = cv2.solvePnP(
                        objPoints, imgPoints, self.matrix_coefficients, self.distortion_coefficients
                    )
                    
                    if success and self.debug:
                        # Draw axis for the marker
                        cv2.drawFrameAxes(frame, self.matrix_coefficients, self.distortion_coefficients, 
                                        rvec, tvec, 0.03)
                        
                        # Display position information
                        marker_position = f"ID:{ids[i][0]} x:{tvec[0][0]:.2f} y:{tvec[1][0]:.2f} z:{tvec[2][0]:.2f}"
                        cv2.putText(frame, marker_position, 
                                (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Print rotation vector for debugging
                        # print(f"Marker {ids[i][0]} rotation vector: {rvec} translation vector: {tvec}")
                        frame = cv2.circle(frame, center=(x_px, y_px), radius=5, color=(0, 255, 0), thickness=-1)
                        
                        cv2.aruco.drawDetectedMarkers(frame, corners, ids)   
            # cv2.imshow("aruco", frame)
            # cv2.waitKey(1)                 
        return frame, rvec, vert
    
    def get_translation_matrix(self, rvec, tvec):
        """
        Get the rotation and translation vectors.
        
        Args:
            rvec: Rotation vector
            tvec: Translation vector
        """

        R, _ = cv2.Rodrigues(rvec)
        tvec4 = np.array([tvec[0], tvec[1], tvec[2]])
        Trans = np.zeros((4, 4), dtype=np.float32)
        Trans[0:3, 0:3] = R.T
        Trans[:-1,3] = R.T @ (-tvec4)
        Trans[3, 3] = 1
        return Trans



    def get_realsense_intrinsics(self):
        
        # Get the first frame
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        # Extract intrinsics from the color stream
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        
        # Create camera matrix
        self.matrix_coefficients = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        
        # Get distortion coefficients
        self.distortion_coefficients = np.array(intrinsics.coeffs)

    def t_camera_to_aruco(self, point):
        """
        Transform a point from camera coordinates to ArUco marker coordinates.
        
        Args:
        point: 3D point in camera coordinates
        Trans: Transformation matrix from camera to ArUco marker
        
        Returns:
        point: Transformed 3D point in ArUco marker coordinates
        """

        point = np.array([point[0], point[1], point[2], 1])
        point = self.translation_matrix @ point
        # point = point[:3] / point[3]
        return point[:-1]


if __name__ == "__main__":
    poses = MediaPipe3DPose(debug=True, translate=True)
    
    while True:
        points = poses.get_arm_points()
        print(points)
