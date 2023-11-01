import tensorflow as tf
import numpy as np
import gpflow
from gpflow.base import TensorType
from gpflow.utilities import positive, print_summary
from check_shapes import inherit_check_shapes


class Fourier(gpflow.kernels.IsotropicStationary):
    def __init__(self, degree: int):
        super().__init__()
        self.variance = gpflow.Parameter(1.0, transform=positive(), name="variance")
        self.cs = gpflow.Parameter([1.0 for _ in range(degree+1)], transform=positive(), name="cs", trainable=False)
        self.degree = degree

    def K_r(self, r: TensorType):
        # r: [batch..., N]
        # deg gives the coefficient inside the cosine
        deg = tf.range(self.degree+1, dtype=tf.float64)
        # match dimensions
        r = tf.expand_dims(r, axis=-1)

        # cs are lengthscales outside of cosine
        c = self.cs
        print(">", tf.math.reduce_sum(tf.cos(deg * r) / 1.0, axis=-1) )
        return self.variance * tf.math.reduce_sum(tf.cos(deg * r) / 1.0, axis=-1)        
