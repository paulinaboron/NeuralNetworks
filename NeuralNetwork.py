import numpy as np
from functions import relu, sigmoid, _linear_forward, relu_backward, sigmoid_backward, _linear_backward
import plotly.express as px


def _forward_propagation(A_prev, W, b, activation):
    """
    Implements the forward propagation for a network layer

    Arguments:
    A_prev -- activations from previous layer, shape : (size of previous layer, number of examples)
    W -- shape : (size of current layer, size of previous layer)
    b -- shape : (size of the current layer, 1)
    activation -- the activation to be used in this layer

    Returns:
    A -- the output of the activation function
    cache -- a python tuple containing "linear_cache" and "activation_cache" for backpropagation
    """

    # Compute Z using the function defined above, compute A using the activaiton function
    if activation == "sigmoid":
        Z, linear_cache = _linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = _linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation == "linear":
        Z, linear_cache = _linear_forward(A_prev, W, b)
        A = Z  # Identity activation for linear
        activation_cache = Z  # Cache Z for backpropagation
        # Store the cache for backpropagation
    cache = (linear_cache, activation_cache)
    return A, cache


def _back_propagation(dA, cache, activation):
    """
    Implements the backward propagation for a single layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache)
    activation -- the activation to be used in this layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    # get the cache from forward propagation and activation derivates function
    linear_cache, activation_cache = cache
    # compute gradients for Z depending on the activation function
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == "linear":
        dZ = dA  # For linear activation, dZ = dA since dA/dZ = 1
    # Compute gradients for W, b and A
    dA_prev, dW, db = _linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


class NeuralNetwork:
    def __init__(self, layer_dimensions=[25, 16, 16, 1], learning_rate=0.01, problem_type="classification", activation_function="relu", initialization_method="he", beta=0.0):
        """
        Parameters
        ----------

        layer_dimensions : list
            python array (list) containing the dimensions of each layer in our network

        learning_rate :  float
            learning rate of the network.

        problem_type : str
            type of problem to solve, either "classification" or "regression".

        activation_function : str
            activation function to use for hidden layers, e.g., "relu", "sigmoid", "linear".

        """

        self.layer_dimensions = layer_dimensions
        self.learning_rate = learning_rate
        self.problem_type = problem_type
        self.activation_function = activation_function
        self.n_layers = len(self.layer_dimensions)
        self.initialization_method = initialization_method
        self.beta = beta

    def initialize_parameters(self):
        for l in range(1, self.n_layers):
            n_prev = self.layer_dimensions[l - 1]

            if self.initialization_method == "he":
                factor = np.sqrt(2. / n_prev)
            elif self.initialization_method == "xavier":
                factor = np.sqrt(1. / n_prev)
            elif self.initialization_method == "small_random":
                factor = 0.01
            else:
                factor = 0.0

            vars(self)[f'W{l}'] = np.random.randn(self.layer_dimensions[l], n_prev) * factor
            vars(self)[f'b{l}'] = np.zeros((self.layer_dimensions[l], 1))

            vars(self)[f'vW{l}'] = np.zeros_like(vars(self)[f'W{l}'])
            vars(self)[f'vb{l}'] = np.zeros_like(vars(self)[f'b{l}'])


    def forward_propagation(self, X):
        """
        Implements forward propagation for the whole network

        Arguments:
        X --  shape : (input size, number of examples)

        Returns:
        AL -- last post-activation value
        caches -- list of cache returned by _forward_propagation helper function
        """
        # Initialize empty list to store caches
        caches = []
        # Set initial A to X
        A = X
        L = self.n_layers - 1
        for l in range(1, L):
            A_prev = A
            # Forward propagate through the network except the last layer
            A, cache = _forward_propagation(A_prev, vars(self)['W' + str(l)], vars(self)['b' + str(l)], self.activation_function)
            caches.append(cache)

        if self.problem_type == "classification":
            activation = "sigmoid"
        else:  # regression
            activation = "linear"

        # Forward propagate through the output layer and get the predictions
        predictions, cache = _forward_propagation(A, vars(self)['W' + str(L)], vars(self)['b' + str(L)], activation)
        # Append the cache to caches list recall that cache will be (linear_cache, activation_cache)
        caches.append(cache)

        return predictions, caches


    def compute_cost(self, predictions, y):
        """
        Implements the cost function

        Arguments:
        predictions -- The model predictions, shape : (1, number of examples)
        y -- The true values, shape : (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """
        if self.problem_type == "classification":
            cost = (-1 / y.shape[0]) * (
                        np.dot(y, np.log(predictions + 1e-9).T) + np.dot((1 - y), np.log(1 - predictions + 1e-9).T))
        else:  # MSE for regression
            y = y.reshape(predictions.shape)
            cost = (1 / (2 * y.shape[1])) * np.sum(np.square(predictions - y))
        return np.squeeze(cost)


    def back_propagation(self, predictions, Y, caches):
        """
        Implements the backward propagation for the NeuralNetwork

        Arguments:
        Prediction --  output of the forward propagation
        Y -- true label
        caches -- list of caches
        """
        L = self.n_layers - 1
        # Get number of examples
        m = predictions.shape[1]
        Y = Y.reshape(predictions.shape)

        # Initializing the backpropagation we're adding a small epsilon for numeric stability
        if self.problem_type == "classification":
            dAL = - (np.divide(Y, predictions + 1e-9) - np.divide(1 - Y, 1 - predictions + 1e-9))
            activation_last = "sigmoid"
        else:  # regresja
            Y = Y.reshape(predictions.shape)
            dAL = (predictions - Y)  # uproszczony gradient dla MSE + linear
            activation_last = "linear"

        current_cache = caches[L - 1]  # Last Layer
        # Compute gradients of the predictions
        vars(self)[f'dA{L - 1}'], vars(self)[f'dW{L}'], vars(self)[f'db{L}'] = _back_propagation(dAL,
                                                                                                      current_cache,
                                                                                                      activation_last)
        for l in reversed(range(L - 1)):
            # update the cache
            current_cache = caches[l]
            # compute gradients of the network layers
            vars(self)[f'dA{l}'], vars(self)[f'dW{l + 1}'], vars(self)[f'db{l + 1}'] = _back_propagation(
                vars(self)[f'dA{l + 1}'], current_cache, activation=self.activation_function)


    def update_parameters(self):
        """
        Updates parameters using gradient descent
        """
        L = self.n_layers - 1
        # Loop over parameters and update them using computed gradients
        for l in range(L):
            vars(self)[f'vW{l + 1}'] = self.beta * vars(self)[f'vW{l + 1}'] + (1 - self.beta) * vars(self)[f'dW{l + 1}']
            vars(self)[f'vb{l + 1}'] = self.beta * vars(self)[f'vb{l + 1}'] + (1 - self.beta) * vars(self)[f'db{l + 1}']

            vars(self)[f'W{l + 1}'] -= self.learning_rate * vars(self)[f'vW{l + 1}']
            vars(self)[f'b{l + 1}'] -= self.learning_rate * vars(self)[f'vb{l + 1}']

    def fit(self, X, Y, epochs=2000, print_cost=True):
        """
        Trains the Neural Network using input data

        Arguments:
        X -- input data
        Y -- true "label"
        Epochs -- number of iterations of the optimization loop
        print_cost -- If set to True, this will print the cost every 100 iterations
        """
        # Transpose X to get the correct shape
        X = X.T
        np.random.seed(1)
        # create empty array to store the costs
        costs = []
        # Get number of training examples
        m = X.shape[1]
        # Initialize parameters
        self.initialize_parameters()
        # loop for stated number of epochs
        for i in range(0, epochs):
            # Forward propagate and get the predictions and caches
            predictions, caches = self.forward_propagation(X)
            # compute the cost function
            cost = self.compute_cost(predictions, Y)
            # Calculate the gradient and update the parameters
            self.back_propagation(predictions, Y, caches)

            self.update_parameters()

            # Print the cost every 10000 training example
            if print_cost and i % 10 == 0:
                costs.append(cost)
        if print_cost:
            # Plot the cost over training
            fig = px.line(y=np.squeeze(costs), title='Cost', template="plotly_dark")
            fig.update_layout(
                title_font_color="#00F1FF",
                xaxis=dict(color="#00F1FF"),
                yaxis=dict(color="#00F1FF")
            )
            fig.show()


    def predict(self, X, y):
        """
        uses the trained model to predict given X value

        Arguments:
        X -- data set of examples you would like to label
        y -- True values of examples; used for measuring the model's accuracy
        Returns:
        predictions -- predictions for the given dataset X
        accuracy or MSE -- depending on problem type
        """
        A, _ = self.forward_propagation(X.T)
        if self.problem_type == "classification":
            predictions = (A > 0.5).astype(int)
            accuracy = np.mean(predictions == y)
            return accuracy, predictions
        else:
            # For regression returning predictions and MSE
            mse = np.mean(np.square(A - y.T))
            return mse, A
