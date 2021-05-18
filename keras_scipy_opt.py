import numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from tqdm.auto import tqdm
import scipy.optimize

from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.ops.gen_array_ops import reshape


def get_loss_and_gradient(model, x_data_array, y_data_array, loss_function, as_numpy=True):
    x_t = tf.convert_to_tensor(x_data_array, dtype=tf.float32)
    y_t = tf.convert_to_tensor(y_data_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        for param in model.trainable_weights:
            tape.watch(param)
        prediction = model(x_t)
        loss = loss_function(y_t, prediction)
        out = loss, tape.gradient(loss, model.trainable_weights)
        if as_numpy:
            out = out[0].numpy(), [t.numpy() for t in out[1]]
    del tape # https://stackoverflow.com/questions/56072634
    return out


def flatten_param_arrays(arrays):
    return np.hstack([(a if isinstance(a, np.ndarray) else a.numpy()).ravel() for a in arrays])

def get_param_vector(model):
    return flatten_param_arrays([t.numpy() for t in model.trainable_weights])

def unflatten_param_vector(model, new_vector):
    assert new_vector.size == model.count_params()
    new_vector = new_vector.ravel()
    intshape = lambda tensor: tuple([int(s) for s in tensor.shape])
    shapes = [intshape(t) for t in model.trainable_weights]
    i1 = 0
    new_arrays = []
    for shape in shapes:
        size = 1
        for s in shape:
            size *= s
        new_arrays.append(new_vector[i1:i1+size].reshape(shape))
        i1 += size
    return new_arrays

def set_param_vector(model, new_vector):
    new_arrays = unflatten_param_vector(model, new_vector)
    return model.set_weights(new_arrays)

def sn(number, digits=2):
    formatter = '%.' + '%d' % digits + 'E'
    factor, exponent = (formatter % number).split('E')
    ffactor = str(float(factor))
    if len(ffactor) < len(factor): factor = ffactor
    exponent = str(int(exponent))
    return r'%s \times 10^{%s}' % (factor, exponent)

def train_sgd(model, x_data, y_data, Loss=tf.keras.losses.MeanSquaredError, batch_size=500, learning_rate=1e-2, n_epochs=2000):
    loss_function = Loss()

    losses = []
    for epoch in tqdm(range(n_epochs)):
        order = np.random.permutation(len(x_data))
        for i in range(0, len(x_data), batch_size):
            j = i+batch_size
            loss, gradients = get_loss_and_gradient(model, x_data[order[i:j]], y_data[order[i:j]], loss_function)
            losses.append(loss)
            params = get_param_vector(model)
            step = - learning_rate * flatten_param_arrays(gradients)
            new_params = params + step
            set_param_vector(model, new_params)

    return losses




def train_scipy(model, x_data, y_data, Loss=tf.keras.losses.MeanSquaredError, **kwargs_scipy):

    loss_function = Loss()

    def checksum(params):
        return np.sum(params)
   
    def objective(parameters):
        set_param_vector(model, parameters)
        loss, gradients = get_loss_and_gradient(model, x_data, y_data, loss_function)
        grad_vec = flatten_param_arrays(gradients)
        objective.last_input_checksum = checksum(parameters)
        objective.last_loss = loss
        objective.last_gradient = grad_vec
        return loss

    def jac(parameters):
        if checksum(parameters) == objective.last_input_checksum:
            return objective.last_gradient
        else:
            objective(parameters)
            return objective.last_gradient

    return scipy.optimize.minimize(objective, get_param_vector(model), jac=jac, **kwargs_scipy)

