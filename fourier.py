"""A Fourier model"""
import matplotlib.pyplot as plt
import numpy as np

import gpflow
import tensorflow as tf
from doubledescent.fourierkernel import Fourier

X = np.array(
    [
        [0.865], [0.666], [0.804], [0.771], [0.147], [0.866], [0.007], [0.026],
        [0.171], [0.889], [0.243], [0.028],
    ]
)
Y = np.array(
    [
        [1.57], [3.48], [3.12], [3.91], [3.07], [1.35], [3.80], [3.82], [3.49],
        [1.30], [4.00], [3.82],
    ]
)

# define model
model = gpflow.models.GPR(
    (X, Y),
    # kernel=gpflow.kernels.SquaredExponential(),
    kernel=Fourier(),
)

# train model
opt = gpflow.optimizers.Scipy()
opt.minimize(model.training_loss, model.trainable_variables)

Xnew = np.array([[0.5]])
model.predict_f(Xnew)
model.predict_y(Xnew)


# save model
model.compiled_predict_f = tf.function(
    lambda Xnew: model.predict_f(Xnew, full_cov=False),
    input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
)
model.compiled_predict_y = tf.function(
    lambda Xnew: model.predict_y(Xnew, full_cov=False),
    input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
)

save_dir = "saved_model_0"
tf.saved_model.save(model, save_dir)

# loaded_model = tf.saved_model.load(save_dir)

# plot_prediction(model.predict_y)
# plot_prediction(loaded_model.compiled_predict_y)


# test points
Xplot = np.linspace(-0.1, 1.1, 100)[:, None]
f_mean, f_var = model.predict_f(Xplot, full_cov=False)
y_mean, y_var = model.predict_y(Xplot)

f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)
y_lower = y_mean - 1.96 * np.sqrt(y_var)
y_upper = y_mean + 1.96 * np.sqrt(y_var)

plt.plot(X, Y, "kx", mew=2, label="input data")
plt.plot(Xplot, f_mean, "-", color="C0", label="mean")
plt.plot(Xplot, f_lower, "--", color="C0", label="f 95% confidence")
plt.plot(Xplot, f_upper, "--", color="C0")
plt.fill_between(
    Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color="C0", alpha=0.1
)
plt.plot(Xplot, y_lower, ".", color="C0", label="Y 95% confidence")
plt.plot(Xplot, y_upper, ".", color="C0")
plt.fill_between(
    Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="C0", alpha=0.1
)
plt.legend()
plt.show()