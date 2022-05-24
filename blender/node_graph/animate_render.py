from easy3d_viewer.graph_visualizer import *
from easy3d_viewer.context import *
import bpy  # Tools provided by blender
import bmesh  # Tools provided by blender
import os
import numpy as np
import mathutils  # Tools provided by blender
import cv2
import sys
import time
from mathutils import Matrix, Vector, Quaternion, Euler  # Tools provided by blender
from mathutils.bvhtree import BVHTree  # Tools provided by blender
import utils.camera_utils

D = bpy.data
C = bpy.context

# Camera projection matrix: (fx, fy, cx, cy);
def get_calibration_matrix_K_from_blender(camd):
    """
    refer to: https://blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
    the code from the above link is wrong, it cause a slight error for fy in "HORIZONTAL" mode or fx in "VERTICAL" mode.
    We did change to fix this.
    """
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == "VERTICAL"):
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
        s_u = s_v/pixel_aspect_ratio
    else: # "HORIZONTAL" and "AUTO"
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = s_u/pixel_aspect_ratio
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px*scale / 2
    v_0 = resolution_y_in_px*scale / 2
    skew = 0 # only use rectangular pixels
    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  ,  alpha_v, v_0),
        (    0  ,    0,      1 )))
    return K

class AnimeRenderInit():
    def __init__(self, config, anime_file=None, dump_path=None, calc_flow=False):
        self.anime_file = anime_file
        self.dump_path = dump_path
        self.calc_flow = calc_flow
        
        # Delete the default cube
        bpy.ops.object.select_all(action="DESELECT")
        if "Cube" in bpy.data.objects.keys():
            bpy.data.objects["Cube"].select_set(state=True)
            bpy.ops.object.delete(use_global=False)
        
        # Set configuration
        self.config = config
        self.cam_locations = np.array(self.config["cam_locations"])
        self.object_center = np.array(self.config["object_center"])
        self.set_camera_property()
    
    def set_camera_property(self, cam_property=None):
        if cam_property is not None:
            pass
        else:
            cam_property = self.config["cam_property"]
        
        bpy_cam = D.objects["Camera"].data
        bpy_cam.angle = eval(cam_property["angle"])
        bpy_scene = bpy.context.scene
        bpy_scene.render.resolution_x = cam_property["image_cols"]
        bpy_scene.render.resolution_y = cam_property["image_rows"]
        bpy.context.view_layer.update()
    
    def run_anime_generator(self, anime_property, cam_idx=0):
        # TODO: Change object_center to automatically defined
        object_center =  np.array(self.config["object_center"])
        z_axis = np.array((0.0, 0.0, 1.0), dtype=np.float32)
        self.world2cam_matrix = utils.camera_utils.look_at_matrix(self.cam_locations[cam_idx], object_center - self.cam_locations[cam_idx], z_axis) # world2camera
        self.cam2world_matrix  = np.linalg.inv(self.world2cam_matrix)
        
        bpy_camera = D.objects["Camera"]
        bpy_camera.matrix_world = self.cam2world_matrix.T
        K = get_calibration_matrix_K_from_blender(bpy_camera.data)
        self.intrinsic = np.array((K[0][0], K[1][1], K[0][2], K[1][2]))
            
        anim = AnimeRenderer(anime_property, self.config, self.dump_path)
        anim.rgbdflowgen(cam_idx)
        outputs = anim.gen_outputs
        outputs["other_property"] = {
            "world2cam": self.world2cam_matrix,
            "cam2world": self.cam2world_matrix,
            "intrinsic": self.intrinsic
        }
        return outputs
    
    def anime_read(self):
        """ Read & parse an .anime file
        Args:
            filename: path of .anime file
        Returns:
            nf: number of frames in the animation
            nv: number of vertices in the mesh (mesh topology fixed through frames)
            nt: number of triangle face in the mesh
            vert_data: vertice data of the 1st frame (3D positions in x-y-z-order)
            face_data: riangle face data of the 1st frame
            offset_data: 3D offset data from the 2nd to the last frame
        """
        f = open(self.anime_file, "rb")
        nf = np.fromfile(f, dtype=np.int32, count=1)[0]
        nv = np.fromfile(f, dtype=np.int32, count=1)[0]
        nt = np.fromfile(f, dtype=np.int32, count=1)[0]
        vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
        face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
        offset_data = np.fromfile(f, dtype=np.float32, count=-1)
        """check data consistency"""
        if len(offset_data) != (nf - 1) * nv * 3:
            raise ("data inconsistent error!", self.anime_file)
        vert_data = vert_data.reshape((-1, 3))
        face_data = face_data.reshape((-1, 3))
        offset_data = offset_data.reshape((nf - 1, nv, 3))

        # # Vis for debug
        # context = Context.Instance()
        # context.addGraph("points")
        # SaveGraph(vertices=vert_data, file_name=context.at("points"))
        return nf, nv, nt, vert_data, face_data, offset_data
    
    
class AnimeRenderer:
    def __init__(self, anime_property, config, dum_path):
        #####################################################################
        self.gen_outputs = {
            "depth": [],
            "rgb": [],
            "flow": [],
            "cam_parameters": {}, 
            "other_property": {}
        }
        
        self.config = config
        self.nf, self.nv, _, vert_data, face_data, offset_data = anime_property
        offset_data = np.concatenate([np.zeros((1, offset_data.shape[1], offset_data.shape[2])), offset_data], axis=0)
        """make object mesh: construct the mesh obj"""
        vertices = vert_data.tolist()
        edges = []
        self.framewise_depth = []
        faces = face_data.tolist()
        mesh_data = bpy.data.meshes.new("mesh_data")
        mesh_data.from_pydata(vertices, edges, faces)
        mesh_data.update()
        the_mesh = bpy.data.objects.new("the_mesh", mesh_data)
        the_mesh.data.vertex_colors.new() # init color
        bpy.context.collection.objects.link(the_mesh)
        #####################################################################
        self.the_mesh = the_mesh
        self.offset_data = offset_data
        self.vert_data = vert_data
        #####################################################################
        self.dum_path = dum_path
        # Stochastical property
        avg_point = np.mean(vert_data, axis=0)
        print(f"Looking center is {avg_point[0], avg_point[1], avg_point[2]}")

    def vis_frame(self, fid):
        """update geometry to a frame (for debug)"""
        src_offset = self.offset_data[fid]
        bm = bmesh.new()  # Control junction
        bm.from_mesh(self.the_mesh.data)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        for i in range(len(bm.verts)):
            bm.verts[i].co = Vector(self.vert_data[i] + src_offset[i])
        bm.to_mesh(self.the_mesh.data)  # Mesh data updated
        bm.free()

    def get_depth_maps(self):
        return np.array(self.framewise_depth)
    
    # Generate RGB-D & scene-flow
    def rgbdflowgen(self, cam_idx = 0, flow_skip=1, render_sflow=True):
        # num_frame = self.offset_data.shape[0]
        outputs_to_file = self.config["outputs_to_file"]
        outputs = self.config["outputs"]
        num_frame = self.nf # # Used for test
        camera = D.objects["Camera"]
        # Parse directory
        if len(outputs_to_file) > 0:
            cam_dump_path = os.path.join(self.dum_path, f"cam-{cam_idx:02}")
            print(f"Saving to {cam_dump_path}")
            if not os.path.exists(cam_dump_path):
                print(f"Creating {cam_dump_path}")
                os.makedirs(cam_dump_path)
            else:
                # Clean previous files
                for file in os.listdir(cam_dump_path):
                    os.remove(os.path.join(cam_dump_path, file))

        #####################################################################
        """prepare rays, (put this inside the for loop if the camera also moves)"""
        K = get_calibration_matrix_K_from_blender(camera.data)
        fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
        width, height = C.scene.render.resolution_x, C.scene.render.resolution_y
        cam_blender = np.array(camera.matrix_world)
        cam_default = utils.camera_utils.blender_to_default(cam_blender)
        u, v = np.meshgrid(range(width), range(height))
        u = u.reshape(-1)
        v = v.reshape(-1)
        pix_position = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1)
        cam_rotation = cam_default[:3, :3]
        pix_position = np.matmul(cam_rotation, pix_position.transpose()).transpose()
        ray_direction = pix_position / np.linalg.norm(pix_position, axis=1, keepdims=True)
        ray_origin = cam_default[:3, 3:].transpose()

        ####################################################################
        """visulize ray geometry(for debug)"""
        vis_ray = False
        if vis_ray:
            ray_end = ray_origin + ray_direction
            ray_vert = np.concatenate([ray_origin, ray_end], axis=0)
            ray_edge = [(0, r_end) for r_end in range(1, len(ray_end) + 1)]
            ray_mesh_data = bpy.data.meshes.new("the_raw")
            ray_mesh_data.from_pydata(ray_vert, ray_edge, [])
            ray_mesh_data.update()
            the_ray = bpy.data.objects.new("the_ray", ray_mesh_data)
            # the_mesh.data.vertex_colors.new()  # init color
            bpy.context.collection.objects.link(the_ray)

        # ####################################################################
        # """dump intrinsics & extrinsics"""
        if "cam_parameters" in outputs:
            self.gen_outputs["cam_parameters"] = {
                "intrinsics": np.array(K),
                "extrinsics": cam_default
            }
        if "cam_parameters" in outputs_to_file:    
            intrin_path = os.path.join(self.dum_path, "cam_intr.txt")
            extrin_path = os.path.join(self.dum_path, "cam_extr.txt")
            np.savetxt (intrin_path, np.array(K))
            np.savetxt (extrin_path, cam_default)

        # #####################################################################
        for src_frame_id in range(num_frame):
            print(f"cam-{cam_idx:02}: {src_frame_id} in {num_frame}")  # Logging
            tgt_frame_id = src_frame_id + flow_skip
            src_offset = self.offset_data[src_frame_id]
            if  tgt_frame_id > num_frame-1 or tgt_frame_id < 0: # video termi
                flow_exist = False
            else :
                flow_exist = True
                tgt_offset = self.offset_data[tgt_frame_id]
                vert_motion = tgt_offset - src_offset  # [N, 3]

            #####################################################################
            """update geometry"""
            bm = bmesh.new()
            bm.from_mesh(self.the_mesh.data)
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            for i in range(len(bm.verts)):
                bm.verts[i].co = Vector(self.vert_data[i] + src_offset[i])
            bm.to_mesh(self.the_mesh.data)
            self.the_mesh.data.update()

            #####################################################################
            """explicitly cast rays to get point cloud and scene flow"""
            # TODO: speedup the code
            # Currently, the code is a bit slower than directly rendering by composition layer of pass_z and pass_uv, (see: https://github.com/lvzhaoyang/RefRESH/tree/master/blender)
            # but since ray_cast return the faceID, this code is more flexible to use, e.g. generating model2frame dense correspondences)
            raycast_mesh = self.the_mesh
            ray_begin_local = raycast_mesh.matrix_world.inverted() @ Vector(ray_origin[0])
            depsgraph = bpy.context.evaluated_depsgraph_get()
            bvhtree = BVHTree.FromObject(raycast_mesh, depsgraph)
            pcl = np.zeros_like(ray_direction)
            rgb = np.zeros([ray_direction.shape[0], 1], np.float32)
            oflow = np.zeros([ray_direction.shape[0], 2], np.float32)  # Optical flow
            for i in range(ray_direction.shape[0]):
                # start = time.time()
                # hit, position, norm, faceID = raycast_mesh.ray_cast(ray_begin_local, Vector(ray_direction[i]), distance=60)
                position, norm, faceID, _ = bvhtree.ray_cast(ray_begin_local, Vector(ray_direction[i]), 50)  # faceID represents the color
                # end = time.time()
                if position: # hit a triangle
                    pcl[i]= Matrix(cam_default).inverted() @ raycast_mesh.matrix_world @ position  # @ is matrix multiplication
                    if ("rgb" in outputs or "rgb" in outputs_to_file) or ("flow" in outputs or "flow" in outputs_to_file):
                        rgb[i] = float(faceID)
                        if render_sflow and flow_exist:
                            face = bm.faces[faceID]
                            vert_index = [ v.index for v in face.verts]
                            vert_vector = [ v.co for v in face.verts ]
                            weights = np.array( mathutils.interpolate.poly_3d_calc (vert_vector, position) )
                            flow_vector = (vert_motion[vert_index] * weights.reshape([3,1])).sum(axis=0)  # Scene-flow
                            # Compute optical flow
                            # Rotate optical flow into camera coordinate
                            flow_vector = np.matmul(np.linalg.inv(cam_default[:3, :3]),  flow_vector.transpose()).transpose() #rotate to camera coordinate system (opencv)
                            # Compute pixel shift
                            oflow_vector = np.array((flow_vector[0] / pcl[i, 2] * fx, flow_vector[1] / pcl[i, 2] * fy))
                            oflow[i, :] = oflow_vector
                            # Try color
                            intp_vector = weights[0] * vert_vector[0] + weights[1] * vert_vector[1] + weights[2] * vert_vector[2]
                            rgb[i] = np.abs(intp_vector[0]) + np.abs(intp_vector[1]) + np.abs(intp_vector[2])
            bm.free()

            #####################################################################
            """dump images"""
            # Depth images
            if ("depth" in outputs or "depth" in outputs_to_file):
                depth = pcl[:, 2].reshape((height, width))
                depth = (depth*1000).astype(np.uint16) #  resolution 1mm
                if "depth" in outputs: 
                    # print(np.nonzero(pcl[:, 2])[0])
                    self.gen_outputs["depth"].append(depth)

                if "depth" in outputs_to_file:
                    depth_path = os.path.join(cam_dump_path, f"frame-{src_frame_id:06}.depth.png")
                    cv2.imwrite(depth_path , depth)
            
            # Rgb images
            if ("rgb" in outputs or "rgb" in outputs_to_file):
                rgb.resize((height, width, 1))
                rgb = (rgb / 3 * 255.0).astype(np.uint8)
                if "rgb" in outputs: 
                    self.gen_outputs["rgb"].append(rgb)
                if "rgb" in outputs_to_file:
                    rgb_map = cv2.applyColorMap(rgb, cv2.COLORMAP_JET)
                    rgb_path = os.path.join(cam_dump_path, f"frame-{src_frame_id:06}.color.png")
                    cv2.imwrite(rgb_path, rgb_map)

            if render_sflow and flow_exist and ("flow" in outputs or "flow" in outputs_to_file):
                # Save opencv file
                flow_path = os.path.join(cam_dump_path, f"frame-{src_frame_id:06}.of.xml")
                flow_file = cv2.FileStorage(flow_path, cv2.FILE_STORAGE_WRITE)
                if "flow" in outputs:
                    self.gen_outputs["flow"].append(oflow)
                if "flow" in outputs_to_file:
                    flow_file.write("optical_flow", oflow)

                # Save img
                oflow.resize((height, width, 2))
                oflow = (oflow / 10.0 * 255.0).astype(np.uint8)
                oflow_x_map = cv2.applyColorMap(oflow[:, :, 0], cv2.COLORMAP_JET)
                flow_x_img_path = os.path.join(cam_dump_path, f"frame-{src_frame_id:06}.ofx.png")
                if "flow" in outputs_to_file:
                    cv2.imwrite(flow_x_img_path, oflow_x_map)

                oflow_y_map = cv2.applyColorMap(oflow[:, :, 1], cv2.COLORMAP_JET)
                flow_y_img_path = os.path.join(cam_dump_path, f"frame-{src_frame_id:06}.ofy.png")
                if "flow" in outputs_to_file:
                    cv2.imwrite(flow_y_img_path, oflow_y_map)