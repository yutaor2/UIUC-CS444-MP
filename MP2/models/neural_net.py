"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
            opt: option for using "SGD" or "Adam" optimizer (Adam is Extra Credit)
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.opt = opt
        
        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])
            
            # TODO: (Extra Credit) You may set parameters for Adam optimizer here
        if self.opt == 'Adam':
            self.m = {}
            self.v = {}
            self.t = 0  
            for i in range(1, self.num_layers + 1):
                self.m["W" + str(i)] = np.zeros_like(self.params["W" + str(i)])
                self.m["b" + str(i)] = np.zeros_like(self.params["b" + str(i)])
                self.v["W" + str(i)] = np.zeros_like(self.params["W" + str(i)])
                self.v["b" + str(i)] = np.zeros_like(self.params["b" + str(i)])


    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        return X.dot(W) + b
    
    def linear_grad(self, W: np.ndarray, X: np.ndarray, de_dz: np.ndarray) -> np.ndarray:
        """Gradient of linear layer
        Parameters:
            W: the weight matrix
            X: the input data
            de_dz: the gradient of loss
        Returns:
            de_dw, de_db, de_dx
            where
                de_dw: gradient of loss with respect to W
                de_db: gradient of loss with respect to b
                de_dx: gradient of loss with respect to X
        """
        dW = X.T.dot(de_dz)
        db = np.sum(de_dz, axis=0)
        dX = de_dz.dot(W.T)
        return dW, db, dX

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        return np.maximum(0, X)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        return np.where(X > 0, 1, 0)


    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # TODO ensure that this is numerically stable
        pos_mask = (x >= 0)
        neg_mask = (x < 0)
        result = np.empty_like(x)
        result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
        exp_x = np.exp(x[neg_mask])
        result[neg_mask] = exp_x / (1 + exp_x)
        return result
    
    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
        return X * (1 - X)

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.mean((y - p) ** 2)
    
    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        N = y.shape[0]
        return 2 * (p - y) / N
    
    def mse_sigmoid_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:                    
        return self.mse_grad(y, p) * self.sigmoid_grad(p)         

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        self.outputs = {}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.
        self.outputs["a0"] = X
        for i in range(1, self.num_layers + 1):
            W = self.params["W" + str(i)]
            b = self.params["b" + str(i)]
            z = self.linear(W, self.outputs["a" + str(i - 1)], b)
            self.outputs["z" + str(i)] = z
            if i < self.num_layers:
                a = self.relu(z)
            else:
                a = self.sigmoid(z)
            self.outputs["a" + str(i)] = a
        return self.outputs["a" + str(self.num_layers)]

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.
        num_layers = self.num_layers
        p = self.outputs["a" + str(num_layers)]
        loss = self.mse(y, p)
        delta = self.mse_sigmoid_grad(y, p)
        for i in range(num_layers, 0, -1):
            a_prev = self.outputs["a" + str(i - 1)]
            W = self.params["W" + str(i)]
            dW, db, dX = self.linear_grad(W, a_prev, delta)
            self.gradients["W" + str(i)] = dW
            self.gradients["b" + str(i)] = db
            if i > 1:
                z_prev = self.outputs["z" + str(i - 1)]
                delta = dX * self.relu_grad(z_prev)
        return loss

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
        """
        if self.opt == 'SGD':
            for i in range(1, self.num_layers + 1):
                self.params["W" + str(i)] -= lr * self.gradients["W" + str(i)]
                self.params["b" + str(i)] -= lr * self.gradients["b" + str(i)]
        elif self.opt == 'Adam':
            self.t += 1
            for i in range(1, self.num_layers + 1):
                for key in ["W", "b"]:
                    param_key = key + str(i)
                    grad = self.gradients[param_key]
                    self.m[param_key] = b1 * self.m[param_key] + (1 - b1) * grad
                    self.v[param_key] = b2 * self.v[param_key] + (1 - b2) * (grad ** 2)
                    m_hat = self.m[param_key] / (1 - (b1 ** self.t))
                    v_hat = self.v[param_key] / (1 - (b2 ** self.t))
                    self.params[param_key] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        else:
            raise NotImplementedError
        