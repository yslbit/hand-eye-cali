import pyzed.sl as sl
import numpy as np
import cv2
from pupil_apriltags import Detector

from utils.coordicate import build_homogeneous

class Cameramanager:
    def __init__(self):
        self.zed = sl.Camera()
        
        self.K = np.array([[536.24, 0, 650.518],
                  [0, 536.21, 359.943],
                  [0, 0, 1]
                  ])  # 相机内参
        
        # 初始化相机
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_units = sl.UNIT.METER
        init_params.sdk_gpu_id = 0
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, 4000)
        
        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            self.zed.close()
            raise RuntimeError("ZED相机打开失败")
            # return None
        print("ZED相机初始化完成")
        
        self.image_zed = sl.Mat()
        self.depth_zed = sl.Mat()
        self.point_cloud = sl.Mat()

        # 初始化AprilTag检测器
        self.detector = Detector(families='tag16h5')

    def get_frame_and_depth(self):
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.depth_zed, sl.MEASURE.DEPTH)
            
            frame = self.image_zed.get_data()
            depth_map = self.depth_zed.get_data()
            return frame.copy(), depth_map.copy()
        return None, None
    
    def get_point_cloud(self):
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
            return self.point_cloud.get_data()
        else:
            print("can't acq point cloud")
            return None
    

    def close(self):
        self.zed.close()
        
    def _extract_apriltag_pose(self, frame):

        K = self.K
        tag_size = 0.05
        tag_family = 'tag16h5'

        
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detection = self.detector.detect(gray_image)

        if detection is None:
            return None, None, None, None
        det = detection[0]

        tag_id = det.tag_id
        
        corners = np.array(det.corners, dtype=np.float32)  # 当前检测器角点顺序已经标准✅
        center = tuple(map(int, det.center))
        
        img_pts = np.array([
            corners[1],
            corners[0],
            corners[3],
            corners[2],
        ], dtype=np.float32)


        # 世界坐标（标准顺序：左上→右上→右下→左下）
        half = tag_size / 2
        obj_pts = np.array([
            [-half,  half, 0],  # 左上
            [ half,  half, 0],  # 右上
            [ half, -half, 0],  # 右下
            [-half, -half, 0],  # 左下
        ], dtype=np.float32)

        # 姿态解算
        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            print(f"solvePnP失败")
            return None, None, None, None
        
        proj_corners, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, None)
        proj_corners = proj_corners.squeeze()
        error_px = np.linalg.norm(proj_corners - img_pts, axis=1)
        mean_error = np.mean(error_px)
        
        if mean_error > 3.0:
            return None, None, None, None
        
        R, _ = cv2.Rodrigues(rvec)
        T_tag2cam = build_homogeneous(R, tvec)
        
        return T_tag2cam, corners, center, mean_error
