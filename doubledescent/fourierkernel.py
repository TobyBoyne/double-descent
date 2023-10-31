import tensorflow as tf
import gpflow
from gpflow.base import TensorType
from gpflow.utilities import positive, print_summary
from check_shapes import inherit_check_shapes


class Fourier(gpflow.kernels.AnisotropicStationary):
    def __init__(self):
        super().__init__()
        self.variance = gpflow.Parameter(1.0, transform=positive())

    @inherit_check_shapes
    def K_d(self, d: TensorType):
        # d: [batch..., N, D]
        dims = tf.range(d.shape[-1], dtype=tf.float64)[None, None, :]
        # add lengthscales later
        c = 0.1
        return self.variance * tf.math.reduce_sum(tf.cos(dims * d) / c, axis=-1)
        
