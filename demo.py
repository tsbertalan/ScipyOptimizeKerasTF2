import numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from keras_scipy_opt import train_sgd, train_scipy, get_loss_and_gradient, sn

x_data = np.linspace(-1, 1, 1000)
true_func = lambda x: x ** 2
y_data = true_func(x_data + np.random.normal(loc=0, scale=.01, size=x_data.shape))


layers = [
    tf.keras.layers.Dense(4, activation='tanh'),
    tf.keras.layers.Dense(4, activation='tanh'),
    tf.keras.layers.Dense(1,  activation='linear'),
]

input_tensor = tf.keras.Input(shape=(1,))
act = input_tensor
for l in layers:
    act = l(act)
output_tensor = act

model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)


if False:
    losses = train_sgd(model, x_data, y_data)

    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel('batch')
    ax.set_ylabel('loss')
    ax.set_yscale('log')

else:
    result = train_scipy(model, x_data, y_data)
    print(result.message)

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
    ax.set_title(r'Loss $\mathcal{L}=%s$' % sn(loss))


show_fit()

plt.show()
