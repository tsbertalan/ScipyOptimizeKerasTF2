"""
Use Scipy's optimization routines to train a simple supervised Keras model.
"""
import numpy as np, tensorflow as tf, matplotlib.pyplot as plt

# Import from our module.
from keras_scipy_opt import (
    train_sgd, train_minimize, train_least_squares, train_curve_fit,
    get_loss_and_gradient, scientific_notation 
)

# Make a stupid little demonstration task.
x_data = np.linspace(-1, 1, 1000)
true_func = lambda x: x ** 2
y_data = true_func(x_data + np.random.normal(loc=0, scale=.01, size=x_data.shape))

# Put together a stadard supervised Keras model.
layers = [
    tf.keras.layers.Dense(4, activation='tanh'),
    tf.keras.layers.Dense(4, activation='tanh'),
    tf.keras.layers.Dense(1,  activation='linear'),
]

# I use the "functional" API rather than the "sequential" because the functional
# includes an Input Tensor, so the shape is pre-specified and therefore the weights
# are initialized.
# However, a Sequential would also work if you did something to get the weights to initialize
# themselves (I guess call the network once with data, and/or call model.compile).
input_tensor = tf.keras.Input(shape=(1,))
act = input_tensor
for l in layers:
    act = l(act)
output_tensor = act

model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

# Train the network deterministically.
# I'll list a few different options here.

# We will not specify any initialization of our own (use what Keras already did).
x0 = None

if False:
    losses = train_sgd(model, x_data, y_data)

    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel('batch')
    ax.set_ylabel('loss')
    ax.set_yscale('log')

# The fastest option seems to be scipy's minimize wrapper.
elif True:
    result = train_minimize(model, x_data, y_data, options=dict(disp=True), method='CG')

# This doesn't seem to converge.
elif False:
    result = train_least_squares(model, x_data, y_data, x0=x0, verbose=2, ftol=1e-8, xtol=1e-8, gtol=1e-8, method='lm')
    print(result.message)

# This does seem to converge, but it's slow.
elif True:
    result = train_least_squares(model, x_data, y_data, x0=x0, verbose=2, method='trf')
    print(result.message)

# This might converge, but it's really slow. I didn't wait.
else:
    result = train_least_squares(model, x_data, y_data, x0=x0, verbose=2, method='dogbox')
    print(result.message)


# Plot the results.
def show_fit():
    fig, ax = plt.subplots()
    ax.scatter(x_data, y_data, color='black', label='Data', alpha=.25, s=1)
    x_smooth = np.linspace(x_data.min(), x_data.max(), 1000)
    ax.plot(x_smooth, true_func(x_smooth), color='black', label='True', linestyle='--')
    y_pred = model.predict(x_smooth)
    ax.plot(x_smooth, y_pred, color='red', label='Network Prediction', linewidth=6, alpha=.5)
    ax.set_xlabel('Inputs $x$')
    ax.set_ylabel('Outputs $y(x)$')
    loss, gl = get_loss_and_gradient(model, x_data, y_data, tf.keras.losses.MeanSquaredError())
    ax.legend()
    ax.set_title(r'Loss $\mathcal{L}=%s$' % scientific_notation(loss))


show_fit()

plt.show()
