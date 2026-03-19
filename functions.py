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
