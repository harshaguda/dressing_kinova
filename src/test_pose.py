import pyrealsense2 as rs
import numpy as np
import cv2

class TestPose():
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        colorizer = rs.colorizer()
        profile = self.pipeline.start(config)
        
        self.frames = self.pipeline.wait_for_frames()
        self.color_frame = self.frames.get_color_frame()
        self.depth_frame = self.frames.get_depth_frame()
        # Extract intrinsics from the color stream
        intrinsics = self.color_frame.profile.as_video_stream_profile().intrinsics
        
        
        # There values are needed to calculate the mapping
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        depth_min = 0.11 #meter
        depth_max = 1.0 #meter

        depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        depth_to_color_extrin =  profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to( profile.get_stream(rs.stream.color))
        color_to_depth_extrin =  profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to( profile.get_stream(rs.stream.depth))


    def get_3d_point_from_pixel(self, depth_frame, color_frame):
        """
        Convert a 2D pixel coordinate (x,y) to a 3D point using the RealSense camera
        
        Args:
            x (int): x-coordinate of the pixel in the image
            y (int): y-coordinate of the pixel in the image
            
        Returns:
            tuple: (x, y, z) coordinates in meters in camera coordinate system
                or None if the depth value at the pixel is invalid
        """
        # print(x, y, dx, dy)
        x, y, dx, dy = 85, 369, 186, 314
        # 53.3 -31.6  31.6
        image = np.asanyarray(color_frame.get_data())
        
        image = cv2.circle(image, center=(x, y), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.imshow("frame", image)
        if cv2.waitKey(1) & 0xFF == 27:
            exit()
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
        print(depth_value, x, y, dx, dy)
        point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth_value)
        return point_3d, depth_value
    

if __name__ == "__main__":
    
    tp = TestPose()
    trans = np.load("translation_matrix.npy")
    
    while True:
        vert, _ = tp.get_3d_point_from_pixel(tp.depth_frame, tp.color_frame)
        P_vert = np.array([vert[0], vert[1], vert[2], 1])
        new_vert = trans @ P_vert
        print("new vert",np.round(new_vert, 3)*100)