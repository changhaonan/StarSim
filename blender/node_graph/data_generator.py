"""
Created by Haonan Chang, 03/28/2022
Note:
- Mask a random number of node during each time-frame.
"""
import os
import math
import numpy as np
import json
import random
import bpy
import sys
dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir )
from utils.graph import *
from utils.spatial_feature import *
from utils.spatial_masker import *
from easy3d_viewer.graph_visualizer import *
from easy3d_viewer.context import *
from pathlib import Path
from animate_render import AnimeRenderInit

world2cam = []
cam2world = []
intrinsic = []

# Reading binary data
def anime_read(filename):
    """
    Author: DeformingThings4D
    filename: path of .anime file
    return:
        nf: number of frames in the animation
        nv: number of vertices in the mesh (mesh topology fixed through frames)
        nt: number of triangle face in the mesh
        vert_data: vertice data of the 1st frame (3D positions in x-y-z-order)
        face_data: riangle face data of the 1st frame
        offset_data: 3D offset data from the 2nd to the last frame
    """
    f = open(filename, 'rb')
    nf = np.fromfile(f, dtype=np.int32, count=1)[0]
    nv = np.fromfile(f, dtype=np.int32, count=1)[0]
    nt = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    '''check data consistency'''
    if len(offset_data) != (nf - 1) * nv * 3:
        raise ("data inconsistent error!", filename)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))
    return nf, nv, nt, vert_data, face_data, offset_data


def look_at_matrix(location, direction, up):
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


class SingleDeformGraphSampler:
    """ There is only one active animation
    """
    def __init__(self, anime_file, config_file, sampling_radius, sampling_mod="Spatial"):
        """
        sampling_mod: "Spatial": unified spaced; "Random": randomly select
        """
        self.sampling_radius = sampling_radius
        self.sampling_mod = sampling_mod
        self.nf, self.nv, self.nt, self.vert_data, self.face_data, self.offset_data = anime_read(anime_file)
        self.anime_file = anime_file
        print('number of frames', self.nf)
        
        with open(config_file, "r") as f:
            self.config = json.loads(f.read())
            
        dump_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../test_data/deer_anime/")
        self.anim_render = AnimeRenderInit(self.config['masking']['animation_render'], anime_file=self.anime_file, dump_path=dump_path)
        self.anime_property = self.anim_render.anime_read()
        self.nf, self.nv, self.nt, self.vert_data, self.face_data, self.offset_data = self.anime_property
        x_max = np.max(self.vert_data[:, 0])
        x_min = np.min(self.vert_data[:, 0])
        y_max = np.max(self.vert_data[:, 1])
        y_min = np.min(self.vert_data[:, 1])
        z_max = np.max(self.vert_data[:, 2])
        z_min = np.min(self.vert_data[:, 2])
        vol_scale = ((x_max - x_min) * (y_max - y_min) * (z_max - z_min)) ** (1.0/3.0)
        vol_scale_inv = 1.0 / vol_scale
        if self.config["enable_model_normalize"]:
            # Rescale model
            self.vert_data = self.vert_data * vol_scale_inv
            self.offset_data = self.offset_data * vol_scale_inv
        else:
            # Rescale sample radius
            # self.sampling_radius = self.sampling_radius * vol_scale
            print("Don't use this for now.")
        
        
    def sample(self):
        """
        spatial_feature should be:
        [T, Nmax, D], where D has
        pos, velocity, t, mask_status
        """
        if self.sampling_mod == "Spatial":
            sampled_v_idx = spatialSampling(self.vert_data, self.sampling_radius)
        else:
            print(f"{self.sampling_mod} is not implemented.")

        sampled_frame = list(range(self.nf - 2))  # Doing no frame sampling currently
        sampled_v = self.vert_data[sampled_v_idx, :]
        n_sample_t = len(sampled_frame)
        n_sample_v = len(sampled_v_idx)
        dim_feature = 3 + 3 + 1
        spatial_feature_array = np.zeros([n_sample_t, n_sample_v, dim_feature], dtype=np.float32)
        for i, t in enumerate(sampled_frame):
            # pos
            sampled_offset = self.offset_data[t, sampled_v_idx, :]
            sampled_pos = sampled_v + sampled_offset  # Use pos in frame t
            # sampled_pos = sampled_v  # Use pos in frame 0
            # velocity
            sampled_velocity = self.offset_data[t + 1, sampled_v_idx, :] - sampled_offset
            # mask_status
            sampled_mask = np.zeros([sampled_pos.shape[0], 1])
            # Concaten
            spatial_feature_array[i, :, :] = np.concatenate([sampled_pos, sampled_velocity, sampled_mask], axis=-1)
        
        # Doing spatial mask
        mask_prob_low = self.config["mask_prob_low"]
        mask_prob_high = self.config["mask_prob_high"]
        mask_prob = np.random.uniform(low=mask_prob_low, high=mask_prob_high)
        voxel_resolution = self.config["voxel_resolution"]
        time_step = self.config["time_step"]
        
        if self.config['masking']['type'] == 'occlusion':
            # dump_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../test_data/deer_anime/")
            # self.anim_render = AnimeRenderInit(self.config['masking']['animation_render'], anime_file=self.anime_file, dump_path=dump_path)
            gen_outputs = self.anim_render.run_anime_generator(self.anime_property)
            depth_maps = gen_outputs['depth']
            spatial_feature_array = SpatialMasker.applyOcclusionMask(spatial_feature_array, depth_maps, gen_outputs['other_props']['world2cam'], gen_outputs['other_props']['intrinsic'])
            world2cam.append(gen_outputs['other_props']['world2cam'])
            cam2world.append(gen_outputs['other_props']['cam2world'])
            intrinsic.append(gen_outputs['other_props']['intrinsic'])
        else:
            spatial_feature_array = SpatialMasker.applyVoxelMask(spatial_feature_array, voxel_resolution, mask_prob, time_step)
        return spatial_feature_array
        

# class MultiDeformGraphSampler:
#     """ Multiple animations are presented in the same scene
#     """
#     def __init__(self, anime_file_list, config_file, force_overlap=False):
#         self.singe_deform_graph_sampler_list = list()
#         for anime_file in anime_file_list:
#             single_deform_graph_sampler = SingleDeformGraphSampler(anime_file, config_file)
#             self.singe_deform_graph_sampler_list.append(single_deform_graph_sampler)
#         self.force_overlap = force_overlap

#     def sample(self):
#         # Take two animation by random
#         [idx1, idx2] = random.sample(range(len(self.singe_deform_graph_sampler_list)), 2)
#         single_sample_1 = self.singe_deform_graph_sampler_list[idx1].sample()
#         single_sample_2 = self.singe_deform_graph_sampler_list[idx2].sample()
        
#         if not self.force_overlap:
#             return np.concatenate([single_sample_1, single_sample_2], axis=1)
#         else:
#             spatial_feature_1 = SpatialFeature(single_sample_1)
#             spatial_feature_2 = SpatialFeature(single_sample_2)
            
#             random_rot_1 = T.Rotation.random()
#             random_rot_2 = T.Rotation.random()
            
#             return np.concatenate([
#                 spatial_feature_1.rotate(random_rot_1).toOrigin().data, 
#                 spatial_feature_2.rotate(random_rot_2).toOrigin().data], axis=1)


class DataGenerator:
    """ Generate data used for training
    """
    def __init__(self, ) -> None:
        # self.dump_path = dump_path
        pass

    def generate(self, anime_file, dump_path, config_file, enable_vis=False):
        """ Generate from anime file to a set of file in dump_path
        """
        # Clean the dump path
        if not os.path.isdir(dump_path):
            os.mkdir(dump_path)
        else:
            for file in os.listdir(dump_path):
                os.remove(os.path.join(dump_path, file))

        config_json = dict()
        with open(config_file, "r") as f:
            config_json = json.loads(f.read())

        deform_graph_sampler = SingleDeformGraphSampler(
            anime_file=anime_file, sampling_radius=config_json["sample_radius_pyramid"][0], config_file=config_file)
        spatial_feature = deform_graph_sampler.sample()

        # Generate graph from sample in the first frame
        vert_pos = spatial_feature[0, :, :3]
        A = buildKNN(vert_pos, 8)
        edge_pair, edge_weight, edge_idx = parseAdjacencyMatrix(A, 8)
        
        # Build graph pyramid
        knn_pyramid = config_json["knn_pyramid"]
        sample_radius_pyramid = config_json["sample_radius_pyramid"]
        pyramid_level = len(knn_pyramid)
        graph_pyramid = buildGraphPyramid(vert_pos, edge_idx, knn_pyramid, sample_radius_pyramid, pyramid_level, enable_vis)

        # Save output
        for t in range(spatial_feature.shape[0]):
            np.save(os.path.join(dump_path, f"{t:04}.npy"), spatial_feature[t])
            saveGraphPyramid(graph_pyramid, os.path.join(dump_path, f"{t:04}.npz"))
        
        # Visualization
        if enable_vis:
            easy3d_viewer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../external/Easy3DViewer")
            print(f"View enabled:")
            print(f"cd {easy3d_viewer_dir}")
            print(f"python configure.py")
            print(f"node app.js")
            print(f"Note: If this your first time, run 'node init' or  'node install' first.")
            anime_name = Path(anime_file).stem
            visualization_dir = anime_name
            context = Context()
            context.setDir(os.path.join(easy3d_viewer_dir, f"public/test_data/{visualization_dir}"), dir_prefix="frame_")
            for i in range(spatial_feature.shape[0]):
                context.open(i)
                context.addGraph(anime_name, size=2, normal_len=1)  # Normal is of scale 1
                if len(world2cam) > 0:
                    context.addCoord("cam", "camera", np.linalg.inv(world2cam[0].T))
                context.addCoord("origin", 'world', coordinate=np.eye(4), scale=1.0)
                SaveGraph(
                    vertices=spatial_feature[i, :, :3],
                    vertex_weight=spatial_feature[i, :, 6:7],
                    edges=edge_pair,
                    normals=spatial_feature[i, :, 3:6],
                    file_name=context.at(anime_name)
                )

                # Pyramid
                vert_level = spatial_feature[i, :, :3]
                for level in range(1, pyramid_level, 1):
                    context.addGraph(f"down_sample_graph{level}", size=2)
                    if len(world2cam) > 0:
                        context.addCoord("cam", "camera", np.linalg.inv(world2cam[0].T))
                    context.addCoord("origin", 'world', coordinate=np.eye(4), scale=1.0)
                    down_sample_idx = graph_pyramid[f"down_sample_idx{level}"]
                    # print(graph_pyramid.keys())
                    vert_level = vert_level[down_sample_idx]
                    SaveGraph(
                        vert_level,
                        vertex_weight=spatial_feature[i, down_sample_idx, 6:7],
                        edges=graph_pyramid[f"nn_index_pair_l{level}"],
                        file_name=context.at(f"down_sample_graph{level}")
                    )
                context.close()


if __name__ == "__main__":
    dump_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test")
    
    import os
    anime_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example.anime")
    # anime_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "foxXAT_Attack1.anime")
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

    data_generator = DataGenerator()
    data_generator.generate(anime_file, dump_path, config_file, True)