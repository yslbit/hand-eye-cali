from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from .coordicate import build_homogeneous
import numpy as np



def rotation_to_vector(rot_mats):
    rots = R.from_matrix(rot_mats)
    
    return rots.as_rotvec()

def cluster_rotation_vectors(rot_vecs, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(rot_vecs)
    return labels, kmeans

def select_representative_samples(kmeans, rotvecs, T_list):
    selected_indices = []
    for i in range(kmeans.n_clusters):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        if len(cluster_indices) == 0:
            continue
        center = kmeans.cluster_centers_[i]
        dists = np.linalg.norm(rotvecs[cluster_indices] - center, axis=1)
        sorted_idx = np.argsort(dists)
        selected = cluster_indices[sorted_idx[[0, -1]] if len(sorted_idx) > 1 else [0]]
        selected_indices.extend(selected.tolist())
    selected_T = [T_list[i] for i in selected_indices]
    return selected_T, selected_indices


    
    
    
    
    
    
    
R_cam2base, t_cam2base = cv2.calibrateHandEye(
                            R_base2end, t_base2end,
                            R_tag2cam, t_tag2cam,
                            method=cv2.CALIB_HAND_EYE_HORAUD)