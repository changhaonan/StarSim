""" Mesh sequence representation
"""
from utils.anime_reader import anime_read
import numpy as np

class MeshSequence: 
    """ General mesh sequence representation
    """
    def __init__(self) -> None:
        self.num_triangle = 0
        self.num_vertex = 0
        self.num_frame = 0
        self.vertex = np.zeros([])
        self.face = np.zeros([]) 
        self.offset = np.zeros([])
        
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
        self.num_triangle = num_triangle
        self.num_vertex = num_vertex
        self.num_frame = num_frame
        self.vertex = vertex
        self.face = face 
        self.offset = offset

    def at(self, t):
        """ Fetch mesh data at frame t

        Args:
            frame (_type_): _description_
        """
        return