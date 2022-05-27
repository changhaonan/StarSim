"""
Created by Haonan Chang, 03/28/2022
Note:
- Major class used for data generation
"""
import os
import sys
import json
import random

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
        self.cam_loc_candidate = self.config["render"]["camera"]["extrinsic_property"]["cam_locations"]
        self.num_cam_loc = len(self.cam_loc_candidate)
        # Init Blender Render
        self.blender_renderer = BlenderRender()
        self.blender_renderer.SetCameraIntrinisc(self.config["render"]["camera"]["intrinsic_property"])
    
    def randCamLoc(self):
        return self.cam_loc_candidate[random.randint(0, self.num_cam_loc - 1)]

    def Generate(self, anime_file, dump_path, enable_vis=False):
        # Generate corresponding data w.r.t anime_file & save to dump_path
        # Mesh sequence
        ms = geometry.mesh_sequence.MeshSequence()
        ms.loadFromAnime(anime_path=anime_file)

        vertex, _, __ = ms.at(0)
        self.blender_renderer.LookAtCenterOfMesh(vertex, self.randCamLoc())
        self.blender_renderer.BindMeshSeqence(ms)
        # Mark visibility
        self.blender_renderer.MarkMeshVisiblity()
        self.blender_renderer.DeBindMeshSequence(ms)

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
    data_generator.Generate(deer_file, dump_deer_path, False)
    data_generator.Generate(bear_file, dump_bear_path, False)