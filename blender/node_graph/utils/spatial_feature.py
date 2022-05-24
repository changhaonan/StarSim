import numpy as np
import scipy.spatial.transform as T

class SpatialFeature:
    """ Data is np.ndarry, use this class for operation
        The structure is:
        [pos(x, y, z), velocity(vx, vy, vz), mask(1/0)]
    """
    def __init__(self, data_array) -> None:
        self.data = data_array
        
    def mask(self):
        return self.data[:, :, 6:7]
    
    def pos(self):
        return self.data[:, :, 0:3]
    
    def velocity(self):
        return self.data[:, :, 3:6]

    def rotate(self, R):
        """ R is a scipy rotation transform
        """
        # Pos is rotated
        rot_vec =  R.apply(self.data[:, :, 0:3].reshape([-1, 3]))
        self.data[:, :, 0:3] = rot_vec.reshape(self.data[:, :, 0:3].shape)
        # Velocity is rotated
        rot_vec =  R.apply(self.data[:, :, 3:6].reshape([-1, 3]))
        self.data[:, :, 3:6] = rot_vec.reshape(self.data[:, :, 3:6].shape)
        return self

    def translate(self, t):
        """ t is (3) vector
        """
        # Pos is translated
        self.data[:, :, 0:3] = self.data[:, :, 0:3] + t.reshape([1, 1, 3])
        return self

    def transform(self, M):
        """ M is a 4*4 matrix
        """
        R = T.Rotation.from_matrix(M[:3, :3])
        t = M[3, :3]
        self.rotate(R)
        self.translate(t)
        return self

    def toOrigin(self):
        """ Set the position mean in each time step to (0, 0, 0)
        """
        origin_offset = np.mean(self.data[:, :, 0:3], axis=1, keepdims=True)
        self.data[:, :, 0:3] = self.data[:, :, 0:3] - origin_offset
        return self

    def getScale(self):
        """ Get the size scale of the feature
        """
        x_max = np.max(self.data[0, :, 0])
        x_min = np.min(self.data[0, :, 0])
        y_max = np.max(self.data[0, :, 1])
        y_min = np.min(self.data[0, :, 1])
        z_max = np.max(self.data[0, :, 2])
        z_min = np.min(self.data[0, :, 2])
        return ((x_max - x_min) * (y_max - y_min) * (z_max - z_min)) ** (1.0/3.0)

    def normalize(self):
        """ resize shape to 1
        """
        scale_inv = 1.0 / self.getScale()
        self.data[:, :, 0:3] = scale_inv * self.data[:, :, 0:3]  # pos
        self.data[:, :, 3:6] = scale_inv * self.data[:, :, 3:6]  # vel
        return self

    def timeLength(self):
        return self.data.shape[0]