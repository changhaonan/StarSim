""" Utils fucntions for visualizing all kinds of data, can be merged to easy3dviewer in the future
"""
import numpy as np
import numpy.matlib as matlib
import open3d as o3d
import cv2

class Visualizer:

    @classmethod
    def DrawDepthImage(cls, depth_img):
        """ Normalize the depth image, and visualize it

        Args:
            depth_img (cv.mat.float32): 2D image with single channel
        """
        cv2.imshow("depth_img", depth_img)
        cv2.waitKey(0)

    @classmethod
    def SaveDepthImage(cls, depth_img, save_path):
        """ Normalize the depth image, and save it

        Args:
            depth_img (cv.mat.float32): 2D image with single channel
            save_path (string): The path where the image will be saved
        """
        norm_image = cv2.normalize(depth_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.imwrite(save_path, norm_image)

    @classmethod
    def SavePointCloud(cls, point_cloud, save_path):
        """ Save the point cloud file into pcd file in ascii, (Compressed is not supported in visualizer)

        Args:
            point_cloud (np.vec3f): array of vetor3f 
            save_path (string): filename where the pcd will be save to
        """
        point_cloud_intensity = np.ones([point_cloud.shape[0], 1])
        if point_cloud.shape[-1] == 3:
            pass
        elif point_cloud.shape[-1] == 4:
            point_cloud_intensity = point_cloud[:, -1]
        else:
            print(f"Input shape {point_cloud.shape} is not accepted.")
        
        point_cloud_position = point_cloud[:, :3]
        pcd = o3d.t.geometry.PointCloud()
        pcd.point["positions"] = o3d.core.Tensor(point_cloud_position)
        # pcd.point["intensities"] = o3d.core.Tensor(point_cloud_intensity)
        o3d.t.io.write_point_cloud(save_path, pcd, True, False)

    @classmethod
    def SavePointCloudWithWeight(cls, point_cloud, point_weight, save_path, weight_low, weight_high):
        """ Save point cloud with corresponding point weights as intensity

        Args:
            point_cloud (np.matNx3): point cloud position
            point_weight (np.matNx1): point cloud weight
            save_path (string): filename where the pcd will be saved to
        """
        pcd = o3d.t.geometry.PointCloud()
        pcd.point["positions"] = o3d.core.Tensor(point_cloud)
        # Compute point color based on point weight
        # point_color = matlib.repmat(point_weight, 1, 3)
        point_color = matlib.repmat(np.array([[1.0, 0.0, 0.0]]), point_weight.shape[0], 1).astype(np.float)
        pcd.point["colors"] = o3d.core.Tensor(point_color)
        o3d.t.io.write_point_cloud(save_path, pcd, True, False)

if __name__ == "__main__":
    import os
    img_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../test_depth.png")
    img = cv2.imread(img_save_path)
    Visualizer.DrawDepthImage(img)