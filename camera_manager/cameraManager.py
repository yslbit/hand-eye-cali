import pyzed.sl as sl
import numpy as np
import cv2
from pupil_apriltags import Detector
import os

from utils.coordicate import build_homogeneous

class Cameramanager:
    def __init__(self, args):
        self.args = args
        self.zed = sl.Camera()
        
        self.K = np.array([[536.24, 0, 650.518],
                  [0, 536.21, 359.943],
                  [0, 0, 1]
                  ])  # 相机内参
        
        self.distCoffs = distCoeffs = np.array([
            -0.05389491,   # k1
            0.02586867,   # k2
            -0.00017095,   # p1
            0.00019722,   # p2
            -0.01096996    # k3
        ])
        
        # init camera
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_units = sl.UNIT.METER
        init_params.sdk_gpu_id = 0
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, 4000)
        
        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            self.zed.close()
            raise RuntimeError("can't open ZED camera")
            # return None
        print("ZED相机初始化完成")
        
        self.image_zed = sl.Mat()
        self.depth_zed = sl.Mat()
        self.point_cloud = sl.Mat()

        # 初始化AprilTag检测器
        self.detector = Detector(families=args.tag)

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
        
    def extract_apriltag_pose(self, frame, get_error=False):

        K = self.K
        tag_size = self.args.tag_size

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        detections = self.detector.detect(gray_image)
        
        if not detections:
            return None, None
        
        det = detections[0]
        
        corners = det.corners   # 逆时针
        img_pts = np.array([corners[0], corners[3], corners[2], corners[1]], dtype=np.float32) # 顺时针

        half = tag_size / 2
        obj_pts = np.array([
            [-half,  half, 0],  # 左上
            [ half,  half, 0],  # 右上
            [ half, -half, 0],  # 右下
            [-half, -half, 0],  # 左下
        ], dtype=np.float32)    # 顺时针
            
        success, rvec, t = cv2.solvePnP(obj_pts, img_pts, K, self.distCoffs, flags=cv2.SOLVEPNP_ITERATIVE) 
        if not success:
            print("solvePnP failed")
            return None, None
        
        # error 
        proj_pts, _ = cv2.projectPoints(obj_pts, rvec, t, K, self.distCoffs)
        proj_pts = proj_pts.squeeze()
        error_px = np.linalg.norm(proj_pts - img_pts, axis=1)
        mean_error = np.mean(error_px)
        
        if get_error:
            return mean_error
        
        self._verify_apriltag_pose(frame, img_pts, proj_pts, rvec, t, K)

        Rmat, _ = cv2.Rodrigues(rvec)
        T_tag2cam = build_homogeneous(Rmat, t)
        
        return T_tag2cam, mean_error
        

    def _verify_apriltag_pose(self, img, img_pts, proj_pts, rvec, t, K, save_path='samples/fig_sampler'):
        
        vis = img.copy()
        
        for i, (pt_img, pt_proj) in enumerate(zip(img_pts, proj_pts)):
            pt_img = tuple(map(int, pt_img))
            pt_proj = tuple(map(int, pt_proj))
            cv2.circle(vis, pt_img, 5, (0, 255, 0), -1)   # 原始角点：绿
            cv2.circle(vis, pt_proj, 5, (255, 0, 0), -1)  # 重投影点：蓝
            cv2.line(vis, pt_img, pt_proj, (0, 255, 255), 1)
            cv2.putText(vis, f"{i}", (pt_img[0] + 5, pt_img[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
        axis_len = self.args.tag_size / 2
        cv2.drawFrameAxes(vis, K, self.distCoffs, rvec, t, axis_len)

        # 自动编号保存图像
        index = self._get_next_index(save_path)
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"vari{index}.jpg")
        cv2.imwrite(file_path, vis)
    
    def _get_next_index(self, save_path, prefix="vari", ext=".jpg"):
        os.makedirs(save_path, exist_ok=True)
        files = os.listdir(save_path)
        indices = []
        for f in files:
            if f.startswith(prefix) and f.endswith(ext):
                try:
                    num = int(f[len(prefix):-len(ext)])
                    indices.append(num)
                except ValueError:
                    continue
        return max(indices, default=-1) + 1
