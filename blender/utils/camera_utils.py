""" Provide utility function for camera, projection and etc
"""
import numpy as np

def blender_to_default(T):
    """ Coordinate in blender is different from default-xyz coordinate

    Args:
        T (np.matrix4f): original transformation in blender

    Returns:
        np.matrix4f: transformation in opencv
    """
    transform = np.array(((1, 0, 0, 0),
              (0, -1, 0, 0),
              (0, 0, -1, 0),
              (0, 0, 0, 1)))
    return np.matmul(T, transform)  #T * transform


def look_at_matrix(location, direction, up):
    """ Assume a camera is looking from location point, looking to 
    direction-axis, with its y-axis aligned with up-axis

    Args:
        location (np.vec3f): where the camera is located
        direction (np.vec3f): where the camera is looking to
        up (np.vec3f): where the y-axis is aligned to

    Returns:
        np.matrix4f: world2cam transformation [Standard]
    """
    z_axis = direction / np.linalg.norm(direction)
    up = up / np.linalg.norm(up)
    x_axis = np.cross(z_axis, up)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(x_axis, z_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = -z_axis
    # Tranformation matrix
    matrix = np.array(
        [
            [x_axis[0], x_axis[1], x_axis[2], -np.dot(x_axis, location)],
            [y_axis[0], y_axis[1], y_axis[2], -np.dot(y_axis, location)],
            [z_axis[0], z_axis[1], z_axis[2], -np.dot(z_axis, location)],
            [0.0, 0.0, 0.0, 1.0]
        ]
    )
    return matrix


def project_vertex_to_depth_image(vertex_array_cam, intrinsic, depth_flip = True):
    """ Project an array of vertex (x, y, z)_i to depth image (u, v, d)_i.

    Args:
        vertex_array_cam (np.matNx3): A set of vertex in camera coordinate
        intrinsic (np.mat2x3): intrinisic matrix
    Returns:
        np.matNx3: The vertex projected to depth image coordinate
    """
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    # print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
    depth_image = np.zeros_like(vertex_array_cam)
    if depth_flip:
        # d = -z
        depth_image[:, 2] = -vertex_array_cam[:, 2]
    else:
        # d = z
        depth_image[:, 2] = vertex_array_cam[:, 2]
    # u = x / z * fx + cx
    depth_image[:, 0] = vertex_array_cam[:, 0] / depth_image[:, 2] * fx + cx
    # v = y / z * fy + cy
    depth_image[:, 1] = vertex_array_cam[:, 1] / depth_image[:, 2] * (-fy) + cy

    return depth_image