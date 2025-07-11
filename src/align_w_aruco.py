import cv2
import numpy as np
import argparse
import pyrealsense2 as rs
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
import cv2

from delta_pose_control import DeltaPoseControl
import numpy as np
import rospy

    
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

translation_matrix  = np.load("/home/userlab/iri_lab/iri_ws/src/dressing_kinova/src/translation_matrix.npy")


def detect_aruco_markers(
        frame, 
        aruco_dict_type,
        matrix_coefficients=None,
        distortion_coefficients=None,
        depth_frame=None,
        color_frame=None,
        depth_colormap=None,
        idx_marker=0,
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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Get ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_dict_type])
    parameters = cv2.aruco.DetectorParameters()
    
    # Detect ArUco markers
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)
    rvec, tvec, vert = None, None, None
    verts = []
    # Draw detected markers if any
    if ids is not None and len(ids) > 0:
        # rvec, tvec, _ = cv2.aruco.(corners, 0.05, matrix_coefficients, distortion_coefficients)
        # print("ids ", ids, len(ids))
        # print(corners)
        for i in range(len(ids)):
            x_px, y_px = int(corners[i].mean(axis=1)[0, 0]), int(corners[i].mean(axis=1)[0, 1])
            vert = get_3d_point_from_pixel(depth_frame, color_frame, x_px, y_px)
            verts.append(t_camera_to_aruco(vert))
            # verts.append(vert)
            # If camera calibration is provided, estimate pose
            if matrix_coefficients is not None and distortion_coefficients is not None:
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
                        objPoints, imgPoints, matrix_coefficients, distortion_coefficients
                    )
                    
                    if success:
                        # Draw axis for the marker
                        # cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, 
                        #                 rvec, tvec, 0.03)
                        
                        # Display position information
                        marker_position = f"ID:{ids[i][0]} x:{tvec[0][0]:.2f} y:{tvec[1][0]:.2f} z:{tvec[2][0]:.2f}"
                        cv2.putText(frame, marker_position, 
                                (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Print rotation vector for debugging
                        # print(f"Marker {ids[i][0]} rotation vector: {rvec} translation vector: {vert}")
            frame = cv2.circle(frame, center=(x_px, y_px), radius=5, color=(i*50, 255, 0), thickness=-1)
            
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    
    return frame, rvec, verts

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
        
        # Get depth value at the pixel (in meters)
        try:
            depth_value = depth_frame.get_distance(x, y)
        except RuntimeError as e:
            print(f"Error getting depth value at pixel ({x}, {y}): {e}")
        if depth_value <= 0:
            print(f"Invalid depth at pixel ({x}, {y})")
        
        # Convert pixel to 3D point in camera coordinate system
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth_value)
        
        return point_3d

def get_realsense_intrinsics():
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    
    # Get the first frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    
    # Extract intrinsics from the color stream
    intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
    
    # Create camera matrix
    camera_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ])
    
    # Get distortion coefficients
    dist_coeffs = np.array(intrinsics.coeffs)
    
    # Stop the pipeline
    pipeline.stop()
    
    return camera_matrix, dist_coeffs

def save_translation_matrix(rvec, tvec):
    """
    Save the rotation and translation vectors to a file.
    
    Args:
        rvec: Rotation vector
        tvec: Translation vector
    """

    R, _ = cv2.Rodrigues(rvec)
    tvec4 = np.array([tvec[0], tvec[1], tvec[2], 1])

    tvec = np.array([tvec[0], tvec[1], tvec[2]])
    Trans = np.zeros((4, 4), dtype=np.float32)
    Trans[0:3, 0:3] = R.T
    Trans[:-1,3] = R.T @ (-tvec)
    Trans[3, 3] = 1
    
    # np.savetxt("translation_matrix.txt", Trans, delimiter=",")
    np.save("translatioqn_matrix", Trans)
    
    print(Trans @ tvec4.T, Trans, tvec4)
    
    return Trans

def t_camera_to_aruco(point):
        """
        Transform a point from camera coordinates to ArUco marker coordinates.
        
        Args:
        point: 3D point in camera coordinates
        Trans: Transformation matrix from camera to ArUco marker
        
        Returns:
        point: Transformed 3D point in ArUco marker coordinates
        """

        point = np.array([point[0], point[1], point[2], 1])
        point = translation_matrix @ point
        # point = point[:3] / point[3]
        return point[:-1]

def angle(elbow, wrist):
    """
    Calculate the angle between the line segment from elbow to wrist and the x-axis."""
    if elbow[0] == wrist[0]:  # Avoid division by zero
        return 0
    if elbow[1] == wrist[1]:  # Horizontal line
        return np.pi / 2
    if elbow[0] == wrist[0] and elbow[1] == wrist[1]:  # Same point
        return 0
    if np.sum(elbow) == 0 or np.sum(wrist) == 0:  # Avoid zero vectors
        return 0

        
    m = (wrist[1] - elbow[1]) / (elbow[0] - wrist[0] )
    return np.arctan(m)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="ArUco Marker Detection")
    parser.add_argument("--dict", type=str, default="DICT_5X5_50", 
                       choices=ARUCO_DICT.keys(), help="ArUco dictionary to use")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    # parser.add_argument("--use_realsense", action="store_true", help="Use Intel RealSense camera")
    parser.add_argument("--calibration", type=str, default=None, help="Camera calibration file (optional)")
    args = parser.parse_args()
    args.use_realsense = True
    # Load camera calibration if provided
    matrix_coefficients = None
    distortion_coefficients = None
    if args.calibration:
        try:
            calib_data = np.load(args.calibration)
            matrix_coefficients = calib_data["camera_matrix"]
            distortion_coefficients = calib_data["dist_coeff"]
            print("Camera calibration loaded successfully")
        except Exception as e:
            print(f"Error loading calibration file: {e}")
    
    # Initialize the camera
    if args.use_realsense:
        try:
            import pyrealsense2 as rs
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            colorizer = rs.colorizer()
            pipeline.start(config)

            
            print("RealSense camera initialized")
        except ImportError:
            print("pyrealsense2 module not found, falling back to webcam")
            cap = cv2.VideoCapture(args.camera)
            args.use_realsense = False
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Error: Could not open camera {args.camera}")
            return
    
    print(f"Using ArUco dictionary: {args.dict}")
    t_pressed = False
    
    trans = np.load("translation_matrix.npy")
    plt.ion()
    prev_angle = 0
    

    dpc = DeltaPoseControl()
    rospy.sleep(5)
    while True:
        # Get frame from camera
        if args.use_realsense:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            # Extract intrinsics from the color stream
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            
            # Create camera matrix
            camera_matrix = np.array([
                [intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
                [0, 0, 1]
            ])
            
            # Get distortion coefficients
            dist_coeffs = np.array(intrinsics.coeffs)
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        else:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break
        
        # Process frame
        output, rvec, vert = detect_aruco_markers(frame, args.dict, 
                                     camera_matrix, dist_coeffs,
                                        depth_frame, color_frame,
                                     depth_colormap)
        if len(vert) == 2:
            angle_z = angle(vert[0], vert[1])
            print(np.rad2deg(angle_z))
            x_plot = [vert[0][0], vert[1][0]]
            y_plot = [vert[0][1], vert[1][1]]
            # x_plot = [0.1, 0.1]
            # y_plot = [0.1, 0.3]
            if abs(prev_angle - angle_z) < 5:
                # print()
                dpc.set_cartesian_pose(x=0, y=0, z=0, tx=0, tz=(90 - np.rad2deg(angle_z)))
            plt.cla()
            plt.plot(x_plot, y_plot)
            plt.xlim([-0.5, 0.5])
            plt.ylim([-0.5, 0.5])
            plt.draw()
            plt.pause(0.1)
            prev_angle = angle_z
        # print(vert)q
        # print(np.rad2deg(angle_z))
        # if vert is not None:
        #     P_vert = np.array([vert[0], vert[1], vert[2], 1])
        #     new_vert = trans @ P_vert
        #     print("new vert",new_vert)
        # Show result
        cv2.imshow("ArUco Marker Detection", output)
        
        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Clean up
    if args.use_realsense:
        pipeline.stop()
    else:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()