from turtle import pos
import numpy as np
import math
import random
from scipy import rand
import cv2
from matplotlib import pyplot as plt

class SpatialMasker:
    """ Apply mask to spatial data
    """
    @classmethod
    def applyVoxelMask(cls, spatial_feature, spatial_resolution, mask_ratio, resample_time_step):
        """ 4D level mask with spatially & temperal voxel mask,

        Args:
            spatial_feature (list[np.vec7f]): num_frame * [pos (3,); vel (3,); mask (1,)]
            spatial_resolution (list3): resoltuion in [dimx, dimy, dimz]
            mask_ratio (float): 
            resample_time_step (int):

        Returns:
            list[np.vec7f]: spatial feature with occlusion mask on the final channel
        """
        mask_ratio = math.pow(mask_ratio, 1.0/3.0)  # Cubic root
        frame = 0
        while frame < spatial_feature.shape[0]:
            # Generate index to be masked
            mask_index_x = random.sample(list(range(spatial_resolution[0])), math.floor(mask_ratio * spatial_resolution[0]))
            mask_index_y = random.sample(list(range(spatial_resolution[1])), math.floor(mask_ratio * spatial_resolution[1]))
            mask_index_z = random.sample(list(range(spatial_resolution[2])), math.floor(mask_ratio * spatial_resolution[2]))

            # Compute pos index
            pos = spatial_feature[frame, :, 0:3]
            # Get x_index
            x_min = np.min(pos[:, 0])
            x_max = np.max(pos[:, 0])
            x_step = (x_max - x_min) / spatial_resolution[0]
            x_index = np.floor((pos[:, 0] - x_min) / x_step)
            mask_x = np.array([False] * pos.shape[0])
            for mask_idx in mask_index_x:
                mask_x = np.logical_or(mask_x, x_index == mask_idx)

            # Get y_index
            y_min = np.min(pos[:, 1])
            y_max = np.max(pos[:, 1])
            y_step = (y_max - y_min) / spatial_resolution[1]
            y_index = np.floor((pos[:, 1] - y_min) / y_step)
            mask_y = np.array([False] * pos.shape[0])
            for mask_idx in mask_index_y:
                mask_y = np.logical_or(mask_y, y_index == mask_idx)

            # Get z_index
            z_min = np.min(pos[:, 2])
            z_max = np.max(pos[:, 2])
            z_step = (z_max - z_min) / spatial_resolution[2]
            z_index = np.floor((pos[:, 2] - z_min) / z_step)
            # print(z_index)
            mask_z = np.array([False] * pos.shape[0])
            for mask_idx in mask_index_z:
                mask_z = np.logical_or(mask_z, z_index == mask_idx)

            mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)
            spatial_feature[frame:frame+resample_time_step, :, 6:7] = mask[:, np.newaxis]
            # Move on
            frame += resample_time_step
        return spatial_feature

    @classmethod
    def applyOcclusionMask(self, spatial_feature, depth_map, world2camera, intrinsic):
        """ Apply a occlusion mask to the spatial feature, occlusion make refers to the occlusion
        we saw in real world.

        Args:
            spatial_feature (list[np.vec7f]): num_frame * [pos (3,); vel (3,); mask (1,)]
            depth_map (cv.mat_uint16): [img_width * img_height]
            world2camera (np.matrix4f): 
            intrinsic (np.vec4f): fx, fy, cx, cy

        Returns:
            list[np.vec7f]: spatial feature with occlusion mask on the final channel
        """
        depths_idx = 0
        frame = 0
        while frame < spatial_feature.shape[0]:
            pos_world = spatial_feature[frame, :, 0:3] ## world coordinate
            depth = depth_map[depths_idx]
            
            # Convert world to camera using world2camera  -> (x, y, z)_c
            pos_world = np.concatenate([pos_world, np.ones((pos_world.shape[0], 1))], axis=-1)
            pos_cam =  world2camera @ pos_world.T
            # Get (u, v)
            pos_cam = pos_cam.T[:, :3]
            fx, fy, cx, cy = intrinsic
            fx = -fx
            pos_pixel = np.concatenate([((np.divide(pos_cam[:, 0], pos_cam[:, 2]) * fx) + cx).reshape(-1, 1), ((np.divide(pos_cam[:, 1], pos_cam[:, 2]) * fy) + cy).reshape(-1, 1)], axis=-1)
            
            # Use this u, v to look for the depths in the depth map, use the z value to check if z >= depth_map(u, v)
            mask = [] 
            grid_img = np.zeros((480, 640))
            for (u, v), z_depth in zip(pos_pixel, pos_cam[:, 2]):
                search_u, search_v = min(479, max(0, math.ceil(v))), min(639, max(0, math.ceil(u)))
                grid_img[search_u, search_v] = 100.
               
                if depth[search_u, search_v] == 0.:
                    mask.append(0.5)
                elif depth[search_u, search_v]/1000.0 < abs(z_depth):
                    mask.append(1)
                else:
                    mask.append(0)
            # print(np.stack([grid_img, depth/1000.], axis=0).shape)
            # plt.imshow((grid_img + depth/1000.))
            # plt.show()
            spatial_feature[frame: frame+1, :, 6:7] = np.array(mask).reshape(1, -1, 1)
            depths_idx += 1
            frame += 1
        return spatial_feature 