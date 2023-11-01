"""Implements a Fourier kernel as described in [Rasmussen and Ghahramani, 2000]
K(x, x') = [\sum_{d=0}^D \cos (d(x-x'))/c_d] / S
"""

import tensorflow as tf
import numpy as np
import gpflow
from gpflow.base import TensorType
from gpflow.utilities import positive, print_summary
from check_shapes import inherit_check_shapes


class Fourier(gpflow.kernels.IsotropicStationary):
    def __init__(self, degree: int, scaling: int):
        super().__init__()
        self.variance = gpflow.Parameter(1.0, transform=positive(), name="variance")
        self.scaling = scaling
        self.degree = degree

    def K_r(self, r: TensorType):
        # r: [batch..., N]
        # deg gives the coefficient inside the cosine
        deg = tf.range(self.degree+1, dtype=tf.float64)
        # match dimensions
        r = tf.expand_dims(r, axis=-1)
        c = tf.pow((deg+1), self.scaling)
        # cs are lengthscales outside of cosine
        return self.variance * tf.math.reduce_sum(tf.cos(deg * r) / c, axis=-1)        
