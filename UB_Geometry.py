import numpy as np
from typing import List, Tuple
import cv2

from cv2 import (
    cvtColor,
    COLOR_BGR2GRAY,
    TERM_CRITERIA_EPS,
    TERM_CRITERIA_MAX_ITER,
    findChessboardCorners,
    cornerSubPix,
    drawChessboardCorners,
)

"""
Please do Not change or add any imports. 
Please do NOT read or write any file, or show any images in your final submission! 
"""

# task1


def findRot_xyz2XYZ(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Args:
        alpha, beta, gamma: They are the rotation angles along x, y and z axis respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from xyz to XYZ.

    """
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    Rz_alpha = np.array(
        [
            [np.cos(alpha_rad), -np.sin(alpha_rad), 0],
            [np.sin(alpha_rad), np.cos(alpha_rad), 0],
            [0, 0, 1],
        ]
    )

    Rx_beta = np.array(
        [
            [1, 0, 0],
            [0, np.cos(beta_rad), -np.sin(beta_rad)],
            [0, np.sin(beta_rad), np.cos(beta_rad)],
        ]
    )

    Rz_gamma = np.array(
        [
            [np.cos(gamma_rad), -np.sin(gamma_rad), 0],
            [np.sin(gamma_rad), np.cos(gamma_rad), 0],
            [0, 0, 1],
        ]
    )

    rot_xyz2XYZ = Rz_alpha @ Rx_beta @ Rz_gamma
    return rot_xyz2XYZ


def findRot_XYZ2xyz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Args:
        alpha, beta, gamma: They are the rotation angles of the 3 step respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from XYZ to xyz.

    """
    rot_xyz2XYZ = findRot_xyz2XYZ(alpha, beta, gamma)
    rot_XYZ2xyz = rot_xyz2XYZ.T

    return rot_XYZ2xyz


"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above "findRot_xyz2XYZ()" and "findRot_XYZ2xyz()" functions are the only 2 function that will be called in task1.py.
"""


# --------------------------------------------------------------------------------------------------------------
# task2:


def find_corner_img_coord(image: np.ndarray) -> np.ndarray:
    gray = cvtColor(image, COLOR_BGR2GRAY)
    H, W = gray.shape
    mid = W // 2
    left, right = gray[:, :mid], gray[:, mid:]

    pattern_size = (4, 4)
    criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.01)

    okL, ptsL = findChessboardCorners(left, pattern_size)
    okR, ptsR = findChessboardCorners(right, pattern_size)

    if okL:
        ptsL = cornerSubPix(left, ptsL, (5, 5), (-1, -1), criteria)
        ptsL[:, 0, 0] += 0.0
    else:
        ptsL = np.zeros((0, 1, 2), dtype=np.float32)

    if okR:
        ptsR = cornerSubPix(right, ptsR, (5, 5), (-1, -1), criteria)
        ptsR[:, 0, 0] += float(mid)
    else:
        ptsR = np.zeros((0, 1, 2), dtype=np.float32)

    pts = np.vstack([ptsL, ptsR]).reshape(-1, 2).astype(float)

    nL = ptsL.shape[0]
    left_pts, right_pts = pts[:nL], pts[nL:]

    def _order_face(p):
        if p.size == 0:
            return np.zeros((0, 2), dtype=float)
        idx = np.lexsort((p[:, 0], p[:, 1]))  # sort by y then x
        return p[idx]

    left_o = _order_face(left_pts)[:16]
    right_o = _order_face(right_pts)[:16]

    img_coord = np.vstack([left_o, right_o]).astype(float)  # 32x2
    return img_coord


def find_corner_world_coord(img_coord: np.ndarray) -> np.ndarray:
    s = 10.0  # mm
    yz = []
    for j in range(4):
        for i in range(4):
            yz.append([0.0, j * s, i * s])
    xz = []
    for j in range(4):
        for i in range(4):
            xz.append([j * s, 0.0, i * s])
    world_coord = np.array(yz + xz, dtype=float)  # 32x3
    return world_coord


def find_intrinsic(
    img_coord: np.ndarray, world_coord: np.ndarray
) -> Tuple[float, float, float, float]:
    # Projection matrix M using DLT
    N = img_coord.shape[0]
    A = []
    for n in range(N):
        X, Y, Z = world_coord[n]
        x, y = img_coord[n]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y])
    A = np.asarray(A, dtype=float)

    _, _, Vt = np.linalg.svd(A)
    m = Vt[-1]
    M = m.reshape(3, 4)
    scale = 1.0 / np.linalg.norm(M[2, :3])
    M = M * scale

    m1, m2, m3 = M[0, :3], M[1, :3], M[2, :3]
    ox = float(m1 @ m3)
    oy = float(m2 @ m3)
    fx = float(np.sqrt(m1 @ m1 - ox * ox))
    fy = float(np.sqrt(m2 @ m2 - oy * oy))
    return fx, fy, ox, oy


def find_extrinsic(
    img_coord: np.ndarray, world_coord: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    # Projection matrix M (DLT)
    N = img_coord.shape[0]
    A = []
    for n in range(N):
        X, Y, Z = world_coord[n]
        x, y = img_coord[n]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y])
    A = np.asarray(A, dtype=float)

    _, _, Vt = np.linalg.svd(A)
    m = Vt[-1]
    M = m.reshape(3, 4)
    scale = 1.0 / np.linalg.norm(M[2, :3])
    M = M * scale

    # Intrinsics
    m1, m2, m3 = M[0, :3], M[1, :3], M[2, :3]
    ox = float(m1 @ m3)
    oy = float(m2 @ m3)
    fx = float(np.sqrt(m1 @ m1 - ox * ox))
    fy = float(np.sqrt(m2 @ m2 - oy * oy))
    K = np.array([[fx, 0.0, ox], [0.0, fy, oy], [0.0, 0.0, 1.0]], dtype=float)

    # Extrinsics
    K_inv = np.linalg.inv(K)
    R_tilde = K_inv @ M[:, :3]
    T = K_inv @ M[:, 3]
    U, _, Vt = np.linalg.svd(R_tilde)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    return R, T


# ---------------------------------------------------------------------------------------------------------------------
