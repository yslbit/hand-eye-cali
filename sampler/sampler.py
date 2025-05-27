import pyzed.sl as sl
import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.pairwise import pairwise_distances
from utils.coordicate import build_homogeneous
from robotcontrol import (Auboi5Robot, RobotError, RobotErrorType, logger_init, logger)
from scipy.optimize import least_squares
import json


class HandEyeCalibrator:
    def __init__(self, robot_api, camera_api, args):
        self.robot = robot_api
        self.camera = camera_api
        self.n_clusters = args.n_clusters
        self.selected_indices = []
        self.T_base2ee = []
        self.T_tag2cam = []
        self.candidate_poses = args.poses_num
        self.current_index = 0
        self.is_kmeans = args.is_kmeans
        self.save_path = args.save_path
        self.max_tag_distance = args.max_tag_distance

        os.makedirs(self.save_path, exist_ok=True)

    def add_sample(self, T_ee, T_tag):
        self.T_base2ee.append(np.linalg.inv(T_ee))
        self.T_tag2cam.append(T_tag)


    def collect_one_sample(self):
        
        frame, depth = self.camera.get_frame_and_depth()
        if frame is None:
            info = f'[ERROR] Camera frame is None, please check the camera connection.'
            print(info)
            return info

        T_ee2base = self._get_current_pose()

        T_tag2cam, _, _, _ = self.camera._extract_apriltag_pose(frame)
        if T_tag2cam is None:
            info = f'[ERROR] AprilTag detection failed, please ensure the tag is visible and properly configured.'
            print(info)
            return info
        
        dist = np.linalg.norm(T_tag2cam[:3, 3])
        if dist > self.max_tag_distance:
            info = f'[ERROR] Calibration board is too far away, distance: {dist:.2f} m, please move it closer.'
            print(info)
            return info
        
        self.add_sample(T_ee2base, T_tag2cam)
        print(f"T_ee2base:\n{T_ee2base}\n")
        self.current_index += 1
        info = f"[INFO] Collected sample {self.current_index}/{self.candidate_poses}."
        print(info)
        return info
        
    def calibrate(self):
        if len(self.T_base2ee) < 3:
            print("[ERROR] Collecting more samples is required for calibration.")
            return None, None

        selected_pairs = None
        if self.is_kmeans:
            selected_pairs = self._select_sample_pose()
            # print(f"selected_pairs: {selected_pairs}")

        T_cam2base, A_list, B_list = self._rigid_motion_solver(selected_pairs)
        print(f"[INFO] Camera to Base T_cam2base:\n{T_cam2base}\n")

        x0 = np.hstack([R.from_matrix(T_cam2base[:3, :3]).as_rotvec().ravel(), T_cam2base[:3, 3]])
        opt = least_squares(self._reproj_error, x0, method='lm', args=(A_list, B_list))
        R_opt = R.from_rotvec(opt.x[:3]).as_matrix()
        t_opt = opt.x[3:6]
        
        print("Optimized Rotation: ", R_opt)
        print("Optimized trans: ", t_opt)

        res = {
            "result":T_cam2base.tolist(),
            "optimized_result": build_homogeneous(R_opt, t_opt).tolist(),
        }
        os.makedirs('./cali_results', exist_ok=True)
        file_name = 'kmeans_result.json' if self.is_kmeans else 'no_kmeans_result.json'
        file_name = os.path.join('./cali_results', file_name)

        with open(file_name, 'w') as f:
            json.dump(res, f)

        return R_opt, t_opt

    def _reproj_error(self, params, A_list: list[np.ndarray], B_list: list[np.ndarray]):
        rvec = params[:3]
        tvec = params[3:6]
        R_x = R.from_rotvec(rvec).as_matrix()
        error = []
        for A, B in zip(A_list, B_list):
            T_x = np.eye(4)
            T_x[:3, :3] = R_x
            T_x[:3, 3] = tvec
            err_mat = A @ T_x - T_x @ B
            error.extend(err_mat[:3, :].flatten())
        return np.array(error)
    
    def _select_sample_pose(self):
        rotvecs = []
        index_pairs = []

        for i in range(len(self.T_base2ee)):
            for j in range(i+1, len(self.T_base2ee)):
                # print(f'self.T_base2ee[i]:\n{self.T_base2ee[i]}\n')
                R1 = self.T_base2ee[i][:3, :3]
                R2 = self.T_base2ee[j][:3, :3]
                R_rel = R.from_matrix(R2 @ R1.T)
                rotvec = R_rel.as_rotvec()

                angle_deg = np.linalg.norm(rotvec) * 180 / np.pi
                if 178 <= angle_deg <= 182:
                    print(f"[SKIP] Relative rotation is close to 180°, non-stablize: {angle_deg:.2f}°，跳过 i={i}, j={j}")
                    continue

                rotvecs.append(rotvec)
                index_pairs.append((i, j))
                        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        labels = kmeans.fit_predict(rotvecs)  # 返回一维数组，表示每个样本的簇标签，代表属于哪个簇
        centers = kmeans.cluster_centers_     # 返回每个簇的中心点

        self.selected_indices = []
        rotvecs = np.stack(rotvecs, axis=0)  # 保存的是相对旋转向量
        labels = np.array(labels) 
        for i in range(self.n_clusters):
            cluster_indices = np.where(labels == i)[0]
            cluster = rotvecs[cluster_indices]
            center = centers[i]
            dists = np.linalg.norm(cluster - center, axis=1)
            idx_center = cluster_indices[np.argmin(dists)]
            idx_farthest = cluster_indices[np.argmax(dists)]

            self.selected_indices.extend([idx_center, idx_farthest])

        selected_pairs = [index_pairs[i] for i in self.selected_indices]

        return selected_pairs

    def _move_to_pose(self, pose):
        
        pos = tuple(pose[:3, 3])
        ori = pose[:3, :3]

        rpy = tuple(R.from_matrix(ori).as_euler('xyz', degrees=True))

        result_type = self.robot.move_to_target_in_cartesian(pos, rpy)
        
        if result_type == RobotErrorType.RobotError_SUCC:
            return True
        else:
            return False
        
    def _rigid_motion_solver(self, pairs=None):
        """Rigid motion solver for hand-eye calibration."""
        T1, T2 = [], []

        def calculate_relative_transformation(i, j):
            """Calculate the relative transformation matrices A and B."""
            # Compute relative transformations between base2ee and tag2cam
            A = np.linalg.inv(self.T_base2ee[j]) @ self.T_base2ee[i]
            B = self.T_tag2cam[j] @ np.linalg.inv(self.T_tag2cam[i])
            return A, B

        # If no pairs are provided, compute for all pairs of transformations
        if pairs is None:
            for i in range(len(self.T_base2ee)):
                for j in range(i + 1, len(self.T_base2ee)):
                    A, B = calculate_relative_transformation(i, j)
                    
                    # Calculate the angle between transformations, and skip if the angle is near 180°
                    angle_deg1 = np.linalg.norm(A) * 180 / np.pi
                    angle_deg2 = np.linalg.norm(B) * 180 / np.pi
                    if 178 <= angle_deg1 <= 182 or 178 <= angle_deg2 <= 182:
                        print(f"[SKIP] Relative rotation angle close to 180°, unstable: {angle_deg1:.2f}, {angle_deg2:.2f}°, skipping i={i}, j={j}")
                        continue
                    
                    T1.append(A)
                    T2.append(B)
        else:
            # If pairs are provided, process only the specified pairs
            for i, j in pairs:
                A, B = calculate_relative_transformation(i, j)
                T1.append(A)
                T2.append(B)

        # Convert rotation matrices to rotation vectors using Rodrigues' formula
        a = [cv2.Rodrigues(x[:3, :3])[0].ravel() for x in T1]
        b = [cv2.Rodrigues(x[:3, :3])[0].ravel() for x in T2]

        # Convert the rotation vectors to numpy arrays for easier manipulation
        b_centered = np.array(b)
        a_centered = np.array(a)

        # Perform Singular Value Decomposition (SVD) to solve for the rotation matrix R
        X = b_centered.T
        Y = a_centered.T
        S = X @ Y.T
        U, _, Vt = np.linalg.svd(S)
        sigma = np.eye(3)
        sigma[-1, -1] = np.linalg.det(Vt.T @ U.T)  # Ensure proper orientation of rotation
        R = Vt.T @ sigma @ U.T

        # Solve for translation vector t using least squares
        A = [x[:3, :3] - np.eye(3) for x in T1]
        B = [R @ x[:3, 3] - y[:3, 3] for x, y in zip(T2, T1)]
        A = np.vstack(A)
        B = np.hstack(B)

        # Solve the linear system for translation t
        t = np.linalg.lstsq(A, B, rcond=None)[0]

        # Compute the final transformation matrix T_cam2base (homogeneous transformation)
        T_cam2base = build_homogeneous(R, t)

        return T_cam2base, T1, T2
    
    def _get_current_pose(self):
        """Get the current pose of the robot end-effector in base frame."""
        try:
            pose = self.robot.get_current_waypoint()
            pos = pose['pos']
            w, x, y, z = pose['ori']

            T_ee2base = np.eye(4)
            T_ee2base[:3, 3] = np.array(pos)
            T_ee2base[:3, :3] = R.from_quat([x, y, z, w]).as_matrix()

            if T_ee2base is None:
                raise RobotError("Failed to get current pose from robot.")
            return T_ee2base
        except RobotError as e:
            print(f"[ERROR] {e}")
            return None

    def load_history_data(self):
        try:
            data_path = os.path.join(self.save_path, 'sample_data.npz')
            data = np.load(data_path)
            
            T_base2ee = data['T_base2ee']
            T_tag2cam = data['T_tag2cam']

            # 确保每个元素是 NumPy 数组并存储为列表
            self.T_base2ee = [T_base2ee[i] for i in range(len(T_base2ee))]
            self.T_tag2cam = [T_tag2cam[i] for i in range(len(T_tag2cam))]
            self.n_clusters = int(data['n_clusters'])

            print(f"T_base2ee type: {type(self.T_base2ee)}")

            self.current_index = len(self.T_base2ee)
            print(f"[INFO] Loaded history data with {len(self.T_base2ee)} samples, n_clusters={self.n_clusters}")
        except FileNotFoundError:
            print(f"[ERROR] {data_path} not found.")

    def save_data(self):
        data_path = os.path.join(self.save_path, 'sample_data.npz')
        if os.path.exists(data_path):
            print(f"[INFO] {data_path} already exists, overwriting...")
        np.savez(data_path, T_base2ee=self.T_base2ee, T_tag2cam=self.T_tag2cam, n_clusters=self.n_clusters)
        print(f"[INFO] Saved {len(self.T_base2ee)} samples to {data_path}")


class RobotManager(Auboi5Robot):
    def __init__(self, ip='192.168.1.1', port=8899, simulator=1):
        logger_init()

        logger.info(f"{Auboi5Robot.get_local_time()} robot init starting...")
        Auboi5Robot.initialize()

        super().__init__()  
        
        self.handle = self.create_context()
        logger.info(f"robot.rshd={self.handle}")

        self.robot_state = self.connect(ip, port)
        
        if self.robot_state != RobotErrorType.RobotError_SUCC:
            logger.info(f"Connect fail {ip}:{port}")
        else:
            self.set_work_mode(mode=simulator)
            self.project_startup()
            self.enable_robot_event()
            self.init_profile()

            joint_maxvelc = (1.5,) * 6
            joint_maxacc = (1,) * 6
            self.set_joint_maxacc(joint_maxacc)
            self.set_joint_maxvelc(joint_maxvelc)

    def disconnect(self):
        if self.connected:
            self.disconnect()
        self.uninitialize()
        
        
        
        
        







