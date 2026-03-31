import numpy as np
import plotly.express as px

def relu(Z):
    """
    Implement the ReLU function.

    Arguments:
    Z -- Output of the linear layer

    Returns:
    A -- Post-activation parameter
    cache -- used for backpropagation
    """
    A = np.maximum(0,Z)
    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single ReLU unit.

    Arguments:
    dA -- post-activation gradient
    cache -- 'Z'  stored for backpropagation

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    # When z <= 0, dz is equal to 0 as well.
    dZ[Z <= 0] = 0
    return dZ


def sigmoid(Z):
    """
    Implement the Sigmoid function.

    Arguments:
    Z -- Output of the linear layer

    Returns:
    A -- Post-activation parameter
    cache -- a python dictionary containing "A" for backpropagation
    """
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single sigmoid unit.

    Arguments:
    dA -- post-activation gradient
    cache -- 'Z' stored during forward pass

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ


def _linear_forward(A, W, b):
    """
    Implements the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- pre-activation parameter
    cache -- a python tuple containing "A", "W" and "b"  for backpropagation
    """
    # Compute Z
    Z = np.dot(W, A) + b
    # Cache  A, W , b for backpropagation
    cache = (A, W, b)
    return Z, cache


def _linear_backward(dZ, cache):
    """
    Implements the linear portion of backward propagation

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output of the current layer
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    # Get the cache from forward propagation
    A_prev, W, b = cache
    # Get number of training examples
    m = A_prev.shape[1]
    # Compute gradients for W, b and A
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def display_function_graph():
    """
    Use this function if you want to see the graphs of ReLU and Sigmoid functions
    """
    z = np.linspace(-12, 12, 200)
    fig = px.line(x=z, y=relu(z)[0], title='ReLU Function', template="plotly_dark")
    fig.update_layout(
        title_font_color="#00F1FF",
        xaxis=dict(color="#00F1FF"),
        yaxis=dict(color="#00F1FF")
    )
    fig.show()

    z = np.linspace(-12, 12, 200)
    fig = px.line(x=z, y=sigmoid(z)[0], title='Sigmoid Function', template="plotly_dark")
    fig.update_layout(
        title_font_color="#00F1FF",
        xaxis=dict(color="#00F1FF"),
        yaxis=dict(color="#00F1FF")
    )
    fig.show()
