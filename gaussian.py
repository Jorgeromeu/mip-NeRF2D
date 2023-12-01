from dataclasses import dataclass

import numpy as np
from scipy.stats import multivariate_normal

@dataclass
class Gaussian2D:
    rotation: float  # rotation in radians
    scale: np.array  # x and y scale
    position: np.array  # x and y pos

    def scipy_gaussian(self):
        return multivariate_normal(self.position, self.covariance_matrix())

    def rot_matrix(self):
        return np.array([
            [np.cos(self.rotation), - np.sin(self.rotation)],
            [np.sin(self.rotation), np.cos(self.rotation)]
        ])

    def scale_matrix(self):
        return np.array([
            [self.scale[0], 0],
            [0, self.scale[1]]
        ])

    def covariance_matrix(self):
        R = self.rot_matrix()
        S = self.scale_matrix()
        return R @ S @ S @ np.linalg.inv(R)
