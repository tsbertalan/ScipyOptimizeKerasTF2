import numpy as np, tensorflow as tf
from tqdm.auto import tqdm
import scipy.optimize

intshape = lambda tensor: tuple([int(s) for s in tensor.shape])

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
    return out

def get_batched_losses(model, x_data_array, y_data_array, loss_function):
    if len(x_data_array.shape) == 1:
        x_data_array = x_data_array.reshape((-1, 1))
    if len(y_data_array.shape) == 1:
        y_data_array = y_data_array.reshape((-1, 1))
    
    N, n_inputs = x_data_array.shape
    assert y_data_array.shape[0] == N
    out = np.empty((N, 1))
    pred = model(x_data_array)
    for i in range(N):
        out[i] = loss_function(y_data_array[i], pred[i])
    return out

def get_batched_losses_and_gradients(model, x_data_array, y_data_array, loss_function):
    if len(x_data_array.shape) == 1:
        x_data_array = x_data_array.reshape((-1, 1))
    if len(y_data_array.shape) == 1:
        y_data_array = y_data_array.reshape((-1, 1))
    
    N, n_inputs = x_data_array.shape
    assert y_data_array.shape[0] == N
    Nparam = model.count_params()
    out = np.empty((N, 1)), [
        np.empty(tuple([N] + list(intshape(t))))
        for t in model.trainable_variables
    ]
    for i in range(N):
        l, gl = get_loss_and_gradient(model, x_data_array[i], y_data_array[i], loss_function, as_numpy=True)
        out[0][i] = l
        for src, dst in zip(gl, out[1]):
            dst[i] = src
    return out

def get_pred_and_jac(model, x_data_array, as_numpy=True):
    x_t = tf.convert_to_tensor(x_data_array.reshape((-1, int(model.inputs[0].shape[1]))), dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_t)
        prediction = model(x_t)
        out = prediction, tape.batch_jacobian(prediction, x_t)
        if as_numpy:
            out = out[0].numpy(), out[1].numpy()
    return out

def flatten_param_arrays(arrays, batched=False):
    if batched:
        N = int(arrays[0].shape[0])
        return np.hstack([
            (a if isinstance(a, np.ndarray) else a.numpy()).reshape((N, -1))
            for a in arrays
        ])
    else:
        return np.hstack([(a if isinstance(a, np.ndarray) else a.numpy()).ravel() for a in arrays])

def get_param_vector(model):
    return flatten_param_arrays([t.numpy() for t in model.trainable_weights])

def unflatten_param_vector(model, new_vector, batched=False):
    if new_vector.size != model.count_params():
        assert len(new_vector.shape) == 2 and batched
        N, nparam = new_vector.shape
    else:
        N = 1
        nparam = new_vector.size
    assert nparam == model.count_params()
    new_vector = new_vector.reshape((N, nparam))
    
    shapes = [intshape(t) for t in model.trainable_weights]
    i1 = 0
    new_arrays = []
    for shape in shapes:
        size = 1
        for s in shape:
            size *= s
        new_arrays.append(new_vector[:, i1:i1+size].reshape(tuple([N] + list(shape))))
        i1 += size
    if not batched:
        return [x[0] for x in new_arrays]
    else:
        return new_arrays

def set_param_vector(model, new_vector):
    new_arrays = unflatten_param_vector(model, new_vector)
    return model.set_weights(new_arrays)

def scientific_notation(number, digits=2):
    """Format a number in scientific notation."""
    formatter = '%.' + '%d' % digits + 'E'
    factor, exponent = (formatter % number).split('E')
    ffactor = str(float(factor))
    if len(ffactor) < len(factor): factor = ffactor
    exponent = str(int(exponent))
    return r'%s \times 10^{%s}' % (factor, exponent)

def train_sgd(model, x_data, y_data, Loss=tf.keras.losses.MeanSquaredError, batch_size=500, learning_rate=1e-2, n_epochs=2000):
    """A simple SGD implementation just to test interfaces."""
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
    """Uses scipy.optimize.minimize; kwargs_scipy (plus jac, the gradient) are passed on to that."""

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

    result = scipy.optimize.minimize(objective, get_param_vector(model), jac=jac, **kwargs_scipy)
    set_param_vector(model, result.x)
    return result

def scipy_curve_fit(model, x_data, y_data, x0=None, verbose=False, **kwargs_scipy):
    """Train with scipy.optimize.curve_fit. Only applicable to scalar-valued models."""

    m = len(x_data)
    if x0 is None:
        x0 = get_param_vector(model)
    n = x0.size

    def f(xdata, *params):
        params = np.array(params)
        set_param_vector(model, params)
        out = model(xdata).numpy()
        if verbose:
            # print('min(residuals)=%s; max(residuals)=%s' % (residuals.min(), residuals.max()), end='; ')
            # print('||residuals|| =', np.linalg.norm(residuals))
            if hasattr(f, 'last_params'):
                step = params - f.last_params
                print('||step|| =', np.linalg.norm(step))
        f.last_params = params
        return out.reshape((xdata.shape[0],))

    popt, pcov = scipy.optimize.curve_fit(f, x_data, y_data, p0=x0, **kwargs_scipy)
    set_param_vector(model, popt)
    return popt

def train_least_squares(model, x_data, y_data, x0=None, **kwargs_scipy):
    """Train with scipy.optimize.least_squares. Not very fast."""

    kwargs_scipy.setdefault('method', 'lm')
    kwargs_scipy.setdefault('ftol', 1e-12)
    kwargs_scipy.setdefault('xtol', 1e-12)
    
    m = len(x_data)
    if x0 is None:
        x0 = get_param_vector(model)
    n = x0.size

    residual_function = lambda truth, prediction: truth - prediction

    def checksum(params):
        return np.sum(params)
   
    def objective(parameters):

        set_param_vector(model, parameters)
        if kwargs_scipy['method'] == 'lm':
            residuals = get_batched_losses(model, x_data, y_data, residual_function)
            verbose = kwargs_scipy.get('verbose', 0)
            if verbose == 2:
                print('min(residuals)=%s; max(residuals)=%s' % (residuals.min(), residuals.max()), end='; ')
                print('||residuals|| =', np.linalg.norm(residuals))
                if hasattr(objective, 'last_params'):
                    step = parameters - objective.last_params
                    print('||step|| =', np.linalg.norm(step))
        else:
            residuals, gradients = get_batched_losses_and_gradients(model, x_data, y_data, residual_function)
            grad_vecs = flatten_param_arrays(gradients, batched=True)
            objective.last_loss = residuals
            objective.last_gradient = grad_vecs

        objective.last_params = parameters
        objective.last_input_checksum = checksum(parameters)

        return residuals.reshape((m,))

    if kwargs_scipy['method'] == 'lm':
        jac = '2-point'
    else:
        def jac(parameters):
            if checksum(parameters) == objective.last_input_checksum:
                return objective.last_gradient
            else:
                objective(parameters)
                return objective.last_gradient

    result = scipy.optimize.least_squares(objective, x0, jac=jac, **kwargs_scipy)
    set_param_vector(model, result.x)
    return result
