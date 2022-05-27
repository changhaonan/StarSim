import numpy as np

def anime_read(anime_file):
    """ Read & parse an .anime file
    Args:
        filename: path of .anime file
    Returns:
        num_frame: number of frames in the animation
        num_vertex: number of vertices in the mesh (mesh topology fixed through frames)
        num_triangle: number of triangle face in the mesh
        vertex: vertice data of the 1st frame (3D positions in x-y-z-order)
        face: riangle face data of the 1st frame
        offset: 3D offset data from the 2nd to the last frame
    """
    f = open(anime_file, "rb")
    num_frame = np.fromfile(f, dtype=np.int32, count=1)[0]
    num_vertex = np.fromfile(f, dtype=np.int32, count=1)[0]
    num_triangle = np.fromfile(f, dtype=np.int32, count=1)[0]
    vertex = np.fromfile(f, dtype=np.float32, count=num_vertex * 3)
    face = np.fromfile(f, dtype=np.int32, count=num_triangle * 3)
    offset = np.fromfile(f, dtype=np.float32, count=-1)
    # check data consistency
    if len(offset) != (num_frame - 1) * num_vertex * 3:
        raise ("data inconsistent error!", anime_file)
    vertex = vertex.reshape((-1, 3))
    face = face.reshape((-1, 3))
    offset = offset.reshape((num_frame - 1, num_vertex, 3))
    
    return num_frame, num_vertex, num_triangle, vertex, face, offset