import torch
import numpy as np

camera2pixelkey = {
    "cam_1": "pixels1",
    "cam_2": "pixels2",
    "cam_51": "pixels51",
}
pixelkey2camera = {v: k for k, v in camera2pixelkey.items()}


def pixel2d_to_3d_torch(points2d, depths, intrinsic_matrix, extrinsic_matrix):
    intrinsic_matrix = torch.tensor(intrinsic_matrix).float().to(depths.device)
    extrinsic_matrix = torch.tensor(extrinsic_matrix).float().to(depths.device)
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    x = (points2d[:, 0] - cx) / fx
    y = (points2d[:, 1] - cy) / fy
    points3d = torch.stack((x * depths, y * depths, depths), dim=1)  # in camera frame
    points3d = torch.cat(
        (points3d, torch.ones((len(points2d), 1)).to(depths.device)), dim=1
    )
    points3d = (torch.linalg.inv(extrinsic_matrix) @ points3d.T).T  # world frame
    return points3d[..., :3]


def pixel2d_to_3d(points2d, depths, intrinsic_matrix, extrinsic_matrix):
    points2d = np.array(points2d)
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    x = (points2d[:, 0] - cx) / fx
    y = (points2d[:, 1] - cy) / fy
    points_3d = np.column_stack((x * depths, y * depths, depths))  # in camera frame
    points_3d = np.concatenate([points_3d, np.ones((len(points2d), 1))], axis=1)
    points_3d = (np.linalg.inv(extrinsic_matrix) @ points_3d.T).T  # world frame
    return points_3d[..., :3]


def pixel3d_to_2d(points3d, intrinsic_matrix, camera_projection_matrix):
    points3d = np.array(points3d)
    points3d = np.concatenate([points3d, np.ones((len(points3d), 1))], axis=1)
    points3d = (camera_projection_matrix @ points3d.T).T  # camera frame
    depth = points3d[:, 2]
    points2d = (intrinsic_matrix @ points3d.T).T
    points2d = points2d / points2d[:, 2][:, None]
    return points2d[..., :2], depth


def triangulate_points(P, points):
    """
    Triangulate a batch of points from a variable number of camera views.

    Parameters:
    P: list of 3x4 projection matrices for each camera (currently world2camera transform)
    points: list of Nx2 arrays of normalized image coordinates for each camera

    Returns:
    Nx4 array of homogeneous 3D points
    """
    num_views = len(P)
    assert num_views > 1, "At least 2 cameras are required for triangulation"

    num_points = points[0].shape[0]
    A = np.zeros((num_points, num_views * 2, 4))

    for idx in range(num_views):
        # Set up the linear system for each point
        A[:, idx * 2] = points[idx][:, 0, np.newaxis] * P[idx][2] - P[idx][0]
        A[:, idx * 2 + 1] = points[idx][:, 1, np.newaxis] * P[idx][2] - P[idx][1]

    # Solve the system using SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[:, -1, :]

    # Normalize the homogeneous coordinates
    X = X / X[:, 3:]

    return X


def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_cols != 3:
        raise Exception(f"matrix A is not Nx3, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_cols != 3:
        raise Exception(f"matrix B is not Nx3, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am.T @ Bm

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A.T + centroid_B.T

    return R, t


def compute_pinch_orientation(hand_point):
    """Estimate the hand's world-frame orientation from index/thumb geometry.

    Used only to seed the frame-0 orientation anchor for the relative
    rigid-transform tracking in convert_pkl_human_to_robot.py — replaces the
    old assumption that the hand starts at identity orientation in the world
    frame.

    Priority #1: the index-tip/thumb-tip vector must exactly match the
    gripper's own finger-separation (Y) axis, since gripper_points.py places
    tip_L/tip_R symmetrically at +-finger_spread along local Y — so setting
    world Y to this vector guarantees the two fingertip-to-fingertip lines
    are exactly parallel. It is used as-is, never adjusted for orthogonality.

    The reach direction (Z) is then derived from the phalange/tip points
    already used above, and orthogonalized against the now-fixed Y axis —
    it absorbs any deviation, rather than the fingertip vector.

    hand_point: (9, 3) array in the point order defined by
    point_policy/point_utils/points_class.py:387-412 —
    0=wrist 1=idx_MCP 2=idx_PIP 3=idx_DIP 4=idx_TIP
    5=thm_CMC 6=thm_MCP 7=thm_IP 8=thm_TIP

    Both `spread` and `forward` are defined with a sign flip (thm->idx
    fingertip rather than idx->thm; tip->phalange rather than phalange->tip):
    robot_base_orientation (a pi rotation about X) flips the Y and Z columns
    of whatever gets composed with it, so both must be pre-negated for the
    final world-frame axes to land in the physically correct direction
    (Y pointing thumb->index, Z pointing away from the wrist toward the
    pinch point, as gripper_points.py's extrapoints assume).
    """
    idx_phalange, idx_tip = hand_point[1], hand_point[4]
    thm_phalange, thm_tip = hand_point[5], hand_point[8]

    spread = thm_tip - idx_tip
    spread = spread / np.linalg.norm(spread)

    forward = (idx_phalange - idx_tip) + (thm_phalange - thm_tip)
    forward = forward - np.dot(forward, spread) * spread  # orthogonalize
    forward = forward / np.linalg.norm(forward)

    lateral = np.cross(spread, forward)
    return np.column_stack([lateral, spread, forward])  # X, Y, Z columns


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """
    Converts 6D rotation representation to rotation matrix
    using Gram-Schmidt orthogonalization.

    Args:
        d6: 6D rotation representation, of shape (..., 6)

    Returns:
        Batch of rotation matrices of shape (..., 3, 3)
    """
    a1, a2 = d6[..., :3], d6[..., 3:]

    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)

    return np.stack((b1, b2, b3), axis=-2)


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """
    Converts rotation matrices to 6D rotation representation
    by dropping the last row.

    Args:
        matrix: Batch of rotation matrices of shape (..., 3, 3)

    Returns:
        6D rotation representation, of shape (..., 6)
    """
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].reshape(batch_dim + (6,))
