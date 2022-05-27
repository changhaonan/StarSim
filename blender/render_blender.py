""" Render method using Blender
"""
import os
import sys
import math
import json
import numpy as np
import numpy.matlib as matlib
import tqdm
import bpy  # Tools provided by blender
import bmesh  # Tools provided by blender
import mathutils
from mathutils import Matrix, Vector, Quaternion, Euler  # Tools provided by blender
from mathutils.bvhtree import BVHTree  # Tools provided by blender

# Pending path
bpy_data_dir = os.path.dirname(bpy.data.filepath)
if not bpy_data_dir in sys.path:
    sys.path.append(bpy_data_dir)
file_data_dir = os.path.dirname(os.path.abspath(__file__))
if not file_data_dir in sys.path:
    print(file_data_dir)
    sys.path.append(file_data_dir)

from utils.blender_utils import get_intrinsic_matrix_from_blender
from utils.camera_utils import blender_to_default, look_at_matrix, project_vertex_to_depth_image

class BlenderRender:
    def __init__(self) -> None:
        # Delete the default cube
        bpy.ops.object.select_all(action="DESELECT")
        if "Cube" in bpy.data.objects.keys():
            bpy.data.objects["Cube"].select_set(state=True)
            bpy.ops.object.delete(use_global=False)

    def SetCameraIntrinisc(self, cam_intrinsic_property):
        """ Set camera intrinsic
        Args:
            cam_intrinsic_property (dict): ["angle", "image_cols", "image_rows"]
        Returns:
            np.matrix23f: intrinsic_matrix
        """
        bpy_cam = bpy.data.objects["Camera"].data
        bpy_cam.angle = eval(cam_intrinsic_property["angle"])
        bpy_scene = bpy.context.scene
        bpy_scene.render.resolution_x = cam_intrinsic_property["image_cols"]
        bpy_scene.render.resolution_y = cam_intrinsic_property["image_rows"]
        bpy.context.view_layer.update()
        self.intrinsic_matrix = get_intrinsic_matrix_from_blender(bpy_cam)
        return self.intrinsic_matrix

    def SetCameraExtrinsic(self, cam2world):
        bpy_cam = bpy.data.objects["Camera"]
        bpy_cam.matrix_world = cam2world.T
        self.extrinsic_matrix = cam2world

    def BindMeshSeqence(self, mesh_sequence):
        """ Bind MeshSequence object to the blender renderer
        Args:
            mesh_sequence (MeshSequence): see utils.mesh_sequence
        """
        # Padding to the length num_frame
        offset = np.concatenate([np.zeros((1, mesh_sequence.offset.shape[1], mesh_sequence.offset.shape[2])), mesh_sequence.offset], axis=0)
        # make object mesh: construct the mesh obj
        vertices = mesh_sequence.vertex.tolist()
        edges = []
        self.framewise_depth = []
        faces = mesh_sequence.face.tolist()
        mesh = bpy.data.meshes.new("mesh")
        mesh.from_pydata(vertices, edges, faces)
        mesh.update()
        the_mesh = bpy.data.objects.new("the_mesh", mesh)
        the_mesh.data.vertex_colors.new() # init color
        bpy.context.collection.objects.link(the_mesh)

        # Bind data
        self.the_mesh = the_mesh
        self.offset = offset
        self.vertex = mesh_sequence.vertex
        self.mesh_sequence = mesh_sequence

    def UpdateFrame(self, t):
        src_offset = self.offset[t]
        bm = bmesh.new()  # Control junction
        bm.from_mesh(self.the_mesh.data)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        for i in range(len(bm.verts)):
            bm.verts[i].co = Vector(self.vertex[i] + src_offset[i])
        bm.to_mesh(self.the_mesh.data)  # Mesh data updated
        bm.free()

    def Render(self, t, render_oflow=False):
        # Build mesh
        bm = bmesh.new()
        bm.from_mesh(self.the_mesh.data)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        for i in range(len(bm.verts)):
            bm.verts[i].co = Vector(self.vertex[i] + self.offset[t][i])
        bm.to_mesh(self.the_mesh.data)
        self.the_mesh.data.update()

        # Explicitly raycast
        # Prepare rays, (put this inside the for loop if the camera also moves)
        camera = bpy.data.objects["Camera"]
        intrinsic_matrix = get_intrinsic_matrix_from_blender(camera.data)
        fx, fy, cx, cy = intrinsic_matrix[0][0], intrinsic_matrix[1][1], intrinsic_matrix[0][2], intrinsic_matrix[1][2]
        width, height = bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y
        cam_blender = np.array(camera.matrix_world)
        cam_default = blender_to_default(cam_blender)
        u, v = np.meshgrid(range(width), range(height))
        u = u.reshape(-1)
        v = v.reshape(-1)
        pix_position = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1)
        cam_rotation = cam_default[:3, :3]
        pix_position = np.matmul(cam_rotation, pix_position.transpose()).transpose()
        ray_direction = pix_position / np.linalg.norm(pix_position, axis=1, keepdims=True)
        ray_origin = cam_default[:3, 3:].transpose()

        # Generate the current depth map #(uint16)
        raycast_mesh = self.the_mesh
        ray_begin_local = raycast_mesh.matrix_world.inverted() @ Vector(ray_origin[0])
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bvhtree = BVHTree.FromObject(raycast_mesh, depsgraph)
        pcl = np.zeros_like(ray_direction)
        oflow = np.zeros([ray_direction.shape[0], 2], np.float32)  # Optical flow
        for i in range(ray_direction.shape[0]):
            position, norm, face_id, _ = bvhtree.ray_cast(ray_begin_local, Vector(ray_direction[i]), 50)  # face_id represents the color
            # end = time.time()
            if position: # hit a triangle
                pcl[i]= Matrix(cam_default).inverted() @ raycast_mesh.matrix_world @ position  # @ is matrix multiplication
                if render_oflow:
                    vert_motion = self.offset[t + 1] - self.offset[t]  # [N, 3]
                    face = bm.faces[face_id]
                    vert_index = [ v.index for v in face.verts ]
                    vert_vector = [ v.co for v in face.verts ]
                    weights = np.array( mathutils.interpolate.poly_3d_calc (vert_vector, position) )
                    flow_vector = (vert_motion[vert_index] * weights.reshape([3,1])).sum(axis=0)  # Scene-flow
                    # Compute optical flow
                    # Rotate optical flow into camera coordinate
                    flow_vector = np.matmul(np.linalg.inv(cam_default[:3, :3]),  flow_vector.transpose()).transpose() # Rotate to camera coordinate system (opencv)
                    # Compute pixel shift
                    oflow_vector = np.array((flow_vector[0] / pcl[i, 2] * fx, flow_vector[1] / pcl[i, 2] * fy))
                    oflow[i, :] = oflow_vector

        # Release mesh    
        bm.free()
        return pcl, pcl[:, 2].reshape((height, width)), oflow.resize((height, width, 2))

    def MarkMeshVisiblity(self):
        """ Generate visible attribute for the mesh sequence that is current binded with the current 
        camera setting
        """
        depth_dev_thresh = 0.01
        image_cols = bpy.context.scene.render.resolution_x
        image_rows = bpy.context.scene.render.resolution_y
        sequence_vertex_visible = np.zeros([self.mesh_sequence.num_frame, self.mesh_sequence.num_vertex, 1])

        for t in range(self.mesh_sequence.num_frame):
            print(f"{self.mesh_sequence.name}: {t}/{self.mesh_sequence.num_frame}.")
            _, depth, __ = self.Render(t)
            vertex, _, __ = self.mesh_sequence.at(t)  # Fetch vertex from mesh sequence
            world2cam = np.linalg.inv(self.extrinsic_matrix)
            vertex_cam = world2cam[0:3, 0:3] @ (vertex.T) + matlib.repmat(world2cam[0:3, 3:4], 1, vertex.shape[0])
            depth_project = project_vertex_to_depth_image(vertex_cam.T, self.intrinsic_matrix, True)
            # Compare with depth & mark
            for i in range(self.mesh_sequence.num_vertex):
                uvd = depth_project[i, :]
                u = math.floor(uvd[0])
                v = math.floor(uvd[1])
                d = uvd[2]
                visible = False
                if u >= 0 and u < image_cols and v >= 0 and v < image_rows:
                    surface_depth = depth[v, u]
                    if abs(surface_depth - d) < depth_dev_thresh:
                        visible = True
                if visible:
                    sequence_vertex_visible[t, i, 0] = 1.0
                else:
                    sequence_vertex_visible[t, i, 0] = 0.0
        print(f"size: {self.mesh_sequence.num_frame * self.mesh_sequence.num_vertex}, visible: {np.count_nonzero(sequence_vertex_visible)}.")
        self.mesh_sequence.addVisiblity(sequence_vertex_visible)


if __name__ == "__main__":
    pass