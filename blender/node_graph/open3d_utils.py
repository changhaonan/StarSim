""" Provide high-level wrapper of operation based on o3d
"""
import numpy as np
import open3d as o3d

def create_point_cloud_from_depth_image(
    depth, intrinsic, extrinisic = np.eye(4), depth_scale=1000.0, depth_trunc=1000.0, stride=1):
    """ Wrapper for o3d.geometry.create_point_cloud_from_depth_image, instead of using open3d 
    object, we are using numpy.array object

    Args:
        depth (np.mat_float32): depth map
        intrinsic (np.vec6f): (fx, fy, cx, cy, width, height)
        extrinisic (np.matrix4f, optional): camera extrinsic. Defaults to np.eye(4).
        depth_scale (float, optional): depth = depth_map / depth_scale. Defaults to 1000.0.
        depth_trunc (float, optional): depth_map = depth_map[depth_map < depth_trunc]. Defaults to 1000.0.
    """
    depth_T = np.ascontiguousarray(depth, dtype=np.float32)
    o3d_depth = o3d.geometry.Image(depth_T)
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    o3d_intrinsic.set_intrinsics(
        int(intrinsic[4]), int(intrinsic[5]), intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3])
    print(np.asarray(o3d_depth).max())
    o3d_pcd = o3d.geometry.PointCloud.create_from_depth_image(
        o3d_depth, 
        o3d_intrinsic, 
        extrinisic, 
        depth_scale, 
        depth_trunc, 
        stride)
    return np.asarray(o3d_pcd.points)