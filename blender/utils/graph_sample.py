""" Sample method
"""
import os
import numpy as np
import json
import utils.graph_utils
from pathlib import Path
import easy3d_viewer.context
import easy3d_viewer.graph_visualizer

class SingleDeformGraphSampler:
    """  Sample a deform graph from a mesh sequence
    """
    def __init__(self, config, sampling_mod="Spatial"):
        """
        sampling_mod: "Spatial": unified spaced; "Random": randomly select
        """
        self.sampling_mod = sampling_mod
        self.config = config
        self.sampling_radius = self.config["sample_radius_pyramid"][0]

    def BindMeshSequence(self, mesh_sequence):
        """ Bind the mesh sequence to the sampler

        Args:
            mesh_sequence (MeshSequence): mesh sequence to sample
        """
        self.mesh_sequence = mesh_sequence
        # Constants
        self.num_frame = mesh_sequence.num_frame
        self.num_vertex = mesh_sequence.num_vertex

        # Resize vertex to scale ~ 1.0
        x_max = np.max(mesh_sequence.vertex[:, 0])
        x_min = np.min(mesh_sequence.vertex[:, 0])
        y_max = np.max(mesh_sequence.vertex[:, 1])
        y_min = np.min(mesh_sequence.vertex[:, 1])
        z_max = np.max(mesh_sequence.vertex[:, 2])
        z_min = np.min(mesh_sequence.vertex[:, 2])
        vol_scale = ((x_max - x_min) * (y_max - y_min) * (z_max - z_min)) ** (1.0/3.0)
        vol_scale_inv = 1.0 / vol_scale

        self.vertex = mesh_sequence.vertex * vol_scale_inv
        self.offset = mesh_sequence.offset * vol_scale_inv

        if mesh_sequence.hasAttribute("visibility"):
            self.visiblity = mesh_sequence.mesh_attribute["visibility"]
        else:
            self.visiblity = np.ones([self.num_frame, self.num_vertex, 1])
        
    def SampleSpatialFeature(self):
        """ Sample a sequence of spatial feature from mesh sequence
        Return:
            spatial_feature_array should be:
            [T, Nmax, D], where D has
            pos, velocity, mask_status
        """
        if self.sampling_mod == "Spatial":
            sampled_v_idx = utils.graph_utils.spatialSampling(self.vertex, self.sampling_radius)
        else:
            print(f"{self.sampling_mod} is not implemented.")

        sampled_frame = list(range(self.num_frame - 2))  # Doing no frame sampling currently
        sampled_v = self.vertex[sampled_v_idx, :]
        n_sample_t = len(sampled_frame)
        n_sample_v = len(sampled_v_idx)
        dim_feature = 3 + 3 + 1
        spatial_feature_array = np.zeros([n_sample_t, n_sample_v, dim_feature], dtype=np.float32)
        for i, t in enumerate(sampled_frame):
            # pos
            sampled_offset = self.offset[t, sampled_v_idx, :]
            sampled_pos = sampled_v + sampled_offset  # Use pos in frame t
            # sampled_pos = sampled_v  # Use pos in frame 0
            # velocity
            sampled_velocity = self.offset[t + 1, sampled_v_idx, :] - sampled_offset
            # mask_status
            visible_mask = self.visiblity[t, sampled_v_idx, :]
            # Concaten
            spatial_feature_array[i, :, :] = np.concatenate([sampled_pos, sampled_velocity, visible_mask], axis=-1)
        
        return spatial_feature_array


    def SampleGraphPyramid(self, dump_path="", enable_vis=False):
        spatial_feature_array = self.SampleSpatialFeature()
        # Generate graph from sample in the first frame
        vert_pos = spatial_feature_array[0, :, :3]
        A = utils.graph_utils.buildKNN(vert_pos, 8)
        edge_pair, edge_weight, edge_idx = utils.graph_utils.parseAdjacencyMatrix(A, 8)
        
        # Build graph pyramid
        knn_pyramid = self.config["knn_pyramid"]
        sample_radius_pyramid = self.config["sample_radius_pyramid"]
        pyramid_level = len(knn_pyramid)
        graph_pyramid = utils.graph_utils.buildGraphPyramid(vert_pos, edge_idx, knn_pyramid, sample_radius_pyramid, pyramid_level, enable_vis)

        # Save output
        if dump_path:
            for t in range(spatial_feature_array.shape[0]):
                np.save(os.path.join(dump_path, f"{t:04}.npy"), spatial_feature_array[t])
                utils.graph_utils.saveGraphPyramid(graph_pyramid, os.path.join(dump_path, f"{t:04}.npz"))

        # Save for visualizatoin
        # Visualization
        if enable_vis:
            easy3d_viewer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../external/Easy3DViewer")
            print(f"View enabled:")
            print(f"cd {easy3d_viewer_dir}")
            print(f"python configure.py")
            print(f"node app.js")
            print(f"Note: If this your first time, run 'node init' or  'node install' first.")
            mesh_name = self.mesh_sequence.name
            visualization_dir = mesh_name
            context = easy3d_viewer.context.Context.Instance()
            context.setDir(os.path.join(easy3d_viewer_dir, f"public/test_data/{visualization_dir}"), dir_prefix="frame_")
            for i in range(spatial_feature_array.shape[0]):
                context.open(i)
                context.addGraph(mesh_name, size=2, normal_len=1)  # Normal is of scale 1
                context.addCoord("origin", 'world', coordinate=np.eye(4), scale=1.0)
                easy3d_viewer.graph_visualizer.SaveGraph(
                    vertices=spatial_feature_array[i, :, :3],
                    vertex_weight=spatial_feature_array[i, :, 6:7],
                    edges=edge_pair,
                    normals=spatial_feature_array[i, :, 3:6],
                    file_name=context.at(mesh_name)
                )

                # Pyramid
                vert_level = spatial_feature_array[i, :, :3]
                for level in range(1, pyramid_level, 1):
                    context.addGraph(f"down_sample_graph{level}", size=2)
                    context.addCoord("origin", 'world', coordinate=np.eye(4), scale=1.0)
                    down_sample_idx = graph_pyramid[f"down_sample_idx{level}"]
                    # print(graph_pyramid.keys())
                    vert_level = vert_level[down_sample_idx]
                    easy3d_viewer.graph_visualizer.SaveGraph(
                        vert_level,
                        edges=graph_pyramid[f"nn_index_pair_l{level}"],
                        file_name=context.at(f"down_sample_graph{level}")
                    )
                context.close()