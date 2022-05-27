""" Mesh sequence representation
"""
from utils.anime_reader import anime_read
import numpy as np
from pathlib import Path

class MeshSequence: 
    """ General mesh sequence representation
    """
    def __init__(self) -> None:
        self.name = "Mesh"
        self.num_triangle = 0
        self.num_vertex = 0
        self.num_frame = 0
        self.vertex = np.zeros([])
        self.face = np.zeros([]) 
        self.offset = np.zeros([])
        # Extendension attribute
        self.mesh_attribute = dict()
        
    def __call__(self, vertex, face, offset) -> None:
        """ Construct from data

        Args:
            vertex (np.vec3f): [num_vertex, 3]
            face (np.vecif): [num_triangle, 3]
            offset (list(np.vec3f)): [num_frame - 1, num_vertex, 3], v_(t+1) - v_t
        """
        self.num_triangle = face.shape[0]
        self.num_vertex = vertex.shape[0]
        self.num_frame = offset.shape[0] + 1
        self.vertex = vertex
        self.face = face
        self.offset = offset

    def loadFromAnime(self, anime_path):
        # Anime file
        num_frame, num_vertex, num_triangle, vertex, face, offset = anime_read(anime_path)
        self.name = Path(anime_path).stem
        self.num_triangle = num_triangle
        self.num_vertex = num_vertex
        self.num_frame = num_frame
        self.vertex = vertex
        self.face = face 
        self.offset = offset

    def at(self, t):
        """ Fetch mesh data at frame t

        Args:
            t (int): time
        Returns:
            np.matNx3: vertex position
            np.matMx3: face data
            dict: attribute
        """
        if t == 0:
            return self.vertex, self.face, self.attributeAt(t)
        elif t <= self.offset.shape[0]:
            return self.vertex + self.offset[t - 1], self.face, self.attributeAt(t)
        else:
            print(f"{t} is out of range: 0~{self.offset.shape[0]}.")
            return np.array([]), np.array([]), dict()

    def attributeAt(self, t):
        mesh_attribute_t = dict()
        for attrib_name in self.mesh_attribute.keys():
            if attrib_name == "visibility":
                mesh_attribute_t["visibility"] = self.mesh_attribute["visibility"][t, :, :]
        return mesh_attribute_t

    def hasAttribute(self, name):
        return name in self.mesh_attribute.keys()

    def addAttribute(self, name, data):
        self.mesh_attribute[name] = data

    def addVisiblity(self, visible_data):
        """ Add visibility attribute to the data

        Args:
            visible_data (list(np.arrayfloat32)): if each vertex is visibility in each time frame
        """
        self.addAttribute("visibility", visible_data)