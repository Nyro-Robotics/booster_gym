import numpy as np


# =============================================================================
# Quaternion operations
# =============================================================================

def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0
    return a - b + c


def quat_rotate_inverse_numpy(q, v):
    """Alias for quat_rotate_inverse for backward compatibility"""
    return quat_rotate_inverse(q, v)


def quat_rotate_numpy(q, v):
    """Quaternion rotation (forward rotation)"""
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0
    # Calculate c
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0

    return a + b + c


def quat_apply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, 1:]
    w = a[:, :1]
    t = np.cross(xyz, b, axis=-1) * 2
    return (b + w * t + np.cross(xyz, t, axis=-1)).reshape(shape)


def quat_apply_yaw(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    quat_yaw = yaw_quat(quat)
    return quat_apply(quat_yaw, vec)


def yaw_quat(quat: np.ndarray) -> np.ndarray:
    quat_yaw = quat.reshape(-1, 4)
    qw = quat_yaw[:, 0]
    qx = quat_yaw[:, 1]
    qy = quat_yaw[:, 2]
    qz = quat_yaw[:, 3]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw = np.zeros_like(quat_yaw)
    quat_yaw[:, 3] = np.sin(yaw / 2)
    quat_yaw[:, 0] = np.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_yaw.reshape(quat.shape)


def quat_xyzw_to_wxyz(q):
    """
    Convert a quaternion from XYZW format to WXYZ format.

    Parameters:
        q (array-like): A quaternion in XYZW format.

    Returns:
        np.ndarray: The quaternion in WXYZ format.
    """
    return np.array([q[3], q[0], q[1], q[2]])


def quat_wxyz_to_xyzw(q):
    """
    Convert a quaternion from WXYZ format to XYZW format.

    Parameters:
        q (array-like): A quaternion in WXYZ format.

    Returns:
        np.ndarray: The quaternion in XYZW format.
    """
    return np.array([q[1], q[2], q[3], q[0]])


def quaternion_to_rotation_matrix(q, w_first=True):
    """
    Convert a quaternion to a 3x3 rotation matrix.

    Parameters:
        q (array-like): A quaternion [w, x, y, z] where:
                        - w is the scalar part
                        - x, y, z are the vector parts

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    if w_first:
        w, x, y, z = q
    else:
        x, y, z, w = q

    # Compute the elements of the rotation matrix
    R = np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )

    return R


def quat_to_rpy(q):
    """
    Convert quaternion to roll, pitch, yaw (ZYX order).
    Input: q = [w, x, y, z]
    Output: roll, pitch, yaw (in radians)
    """
    w, x, y, z = q

    # Roll (X-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (Y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (Z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def rpy_to_quat(rpy):
    """
    Convert roll, pitch, yaw (in radians) to quaternion [w, x, y, z]
    Follows ZYX rotation order (yaw → pitch → roll)
    """
    roll, pitch, yaw = rpy
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


# =============================================================================
# Vector and matrix operations
# =============================================================================

def normalize(x, eps: float = 1e-9):
    """Normalize a vector to unit length"""
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=eps, max=None)


def normalize_range(x, min_value, max_value, target_min=0.0, target_max=1.0):
    """
    Normalize a value from a given range to a target range.
    """
    # Normalize the value to the range [0, 1]
    normalized = (x - min_value) / (max_value - min_value)

    # Scale the value to the target range
    scaled = target_min + normalized * (target_max - target_min)

    return scaled


def unnormalize(x, min_value, max_value, target_min=0.0, target_max=1.0):
    """
    Unnormalize a value from a target range to a given range.
    """
    # Normalize the value to the range [0, 1]
    normalized = (x - target_min) / (target_max - target_min)

    # Scale the value to the target range
    scaled = min_value + normalized * (max_value - min_value)

    return scaled


def skew_symmetric(p):
    """
    Generate a skew-symmetric matrix from a 3D vector.

    Parameters:
        p (array-like): A 3D vector (list, tuple, or NumPy array) of length 3.

    Returns:
        np.ndarray: A 3x3 skew-symmetric matrix.
    """
    if len(p) != 3:
        raise ValueError("Input vector must have exactly 3 elements.")
    return np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])


# =============================================================================
# Angle operations
# =============================================================================

def wrap_to_pi(angles):
    """Wrap angles to the range [-π, π]"""
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles
