"""
Created by Haonan Chang, 03/28/2022
Note:
- Major class used for data generation
"""
import os
import sys
import json
import numpy as np

file_data_dir = os.path.dirname(os.path.abspath(__file__))
if not file_data_dir in sys.path:
    print(file_data_dir)
    sys.path.append(file_data_dir)

import geometry.mesh_sequence
from render_blender import BlenderRender
import utils.camera_utils
import utils.graph_sample

class DataGeneratorNodeFlow:
    """ Wrapper DataGenerator for NodeFlow
    """
    def __init__(self, config_path) -> None:
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        f = open(config_file)
        self.config = json.load(f)

    def Generate(self, anime_file, dump_path, enable_vis=False):
        # Generate corresponding data w.r.t anime_file & save to dump_path
        # Mesh sequence
        ms = geometry.mesh_sequence.MeshSequence()
        ms.loadFromAnime(anime_path=anime_file)

        # Init Blender Render
        blender_renderer = BlenderRender()
        intrinsic_matrix = blender_renderer.SetCameraIntrinisc(self.config["render"]["camera"]["intrinsic_property"])
        image_rows = self.config["render"]["camera"]["intrinsic_property"]["image_rows"]
        image_cols = self.config["render"]["camera"]["intrinsic_property"]["image_cols"]
        # Get look at matrix
        cam_pos = np.array(self.config["render"]["camera"]["extrinsic_property"]["cam_locations"][0])
        obj_center = np.array(self.config["render"]["camera"]["extrinsic_property"]["object_center"])
        world2cam = utils.camera_utils.look_at_matrix(
            cam_pos,
            obj_center - cam_pos,
            np.array([0, 0, 1])
        )
        cam2world = np.linalg.inv(world2cam)
        blender_renderer.SetCameraExtrinsic(cam2world)
        blender_renderer.BindMeshSeqence(ms)

        # Mark visibility
        blender_renderer.MarkMeshVisiblity()

        # Generate graph praymid
        single_graph_sampler = utils.graph_sample.SingleDeformGraphSampler(self.config)
        single_graph_sampler.BindMeshSequence(ms)
        single_graph_sampler.SampleGraphPyramid(dump_path, enable_vis)


if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    dump_deer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output/deer")
    dump_bear_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output/bear")
    deer_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data/deer.anime")
    bear_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data/bear.anime")
    data_generator = DataGeneratorNodeFlow(config_file)
    data_generator.Generate(deer_file, dump_deer_path, True)
    data_generator.Generate(bear_file, dump_bear_path, True)