import numpy as np

def build_homogeneous(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array(t).flatten()
    return T