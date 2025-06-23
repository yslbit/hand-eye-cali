import pyzed.sl as sl
import numpy as np
import cv2
from pupil_apriltags import Detector

from utils.coordicate import build_homogeneous

class Cameramanager:
    def __init__(self, args):
        self.args = args
        self.zed = sl.Camera()
        
        self.K = np.array([[536.24, 0, 650.518],
                  [0, 536.21, 359.943],
                  [0, 0, 1]
                  ])  # 相机内参
        
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
        
        corners = np.array(det.corners, dtype=np.float32)
        center = tuple(map(int, det.center))
        
        img_pts = self._sort_corners_clockwise(corners)


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

        self._verify_pnp_results(frame, obj_pts, img_pts, rvec, tvec, K, draw=True)
        
        return T_tag2cam, corners, center, mean_error
    
    def _sort_corners_clockwise(self, corners):
        
        """
        Sort corners in clockwise order starting from the top-left corner.
        Args:
            corners (np.ndarray): Array of shape (4, 2) containing the corners of the tag.
        Returns:
            np.ndarray: Sorted corners in clockwise order.
            left top, right top, right bottom, left bottom
        
        """
        # Calculate the centroid of the corners
        corners = np.array(corners, dtype=np.float32)

        center = np.mean(corners, axis=0)

        def angle_from_center(point):
            delta = point - center
            return np.arctan2(delta[1], delta[0])

        # Sort corners by angle from the center
        sorted_pts = sorted(corners, key=angle_from_center)
        sorted_pts = np.array(sorted_pts, dtype=np.float32)

        top_left_idx =  np.argmin(sorted_pts[:, 0] + sorted_pts[:, 1])

        sorted_pts = np.roll(sorted_pts, -top_left_idx, axis=0)

        return sorted_pts

    def _verify_pnp_results(self, img, obj_pts, img_pts, rvec, tvec, camera_matrix, draw=True, save_path='samples/fig_sampler/pnp_verification.jpg'):
        """
        Verify the PnP results by projecting the object points back to image space
        and checking the reprojection error.
        """
        proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, None)

        proj_pts = proj_pts.squeeze()
        img_pts = img_pts.squeeze()

        error = np.linalg.norm(proj_pts - img_pts, axis=1)
        mean_error = np.mean(error)

        if draw:
            img = img.copy()
            for i, (pt_img, pt_proj) in enumerate(zip(img_pts, proj_pts)):
                pt_img = tuple(map(int, pt_img))
                pt_proj = tuple(map(int, pt_proj))

                cv2.circle(img, pt_img, 5, (0, 255, 0), -1)
                cv2.circle(img, pt_proj, 5, (255, 0, 0), -1)

                cv2.putText(img, f"{i}", (pt_img[0]+5, pt_img[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                cv2.line(img, pt_img, pt_proj, (0, 255, 255), 1)

            cv2.imwrite(save_path, img)

        return mean_error
