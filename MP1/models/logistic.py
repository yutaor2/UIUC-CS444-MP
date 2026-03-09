"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.mean = None
        self.std = None
        
    def normalize(self, X: np.ndarray, is_train: bool = True) -> np.ndarray:
        if is_train:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        return (X - self.mean) / (self.std + 1e-8)


    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        # Hint: To prevent numerical overflow, try computing the sigmoid for positive numbers and negative numbers separately.
        #       - For negative numbers, try an alternative formulation of the sigmoid function.
        result = np.empty_like(z)
        pos_mask = z >= 0
        result[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))
        result[~pos_mask] = np.exp(z[~pos_mask]) / (1 + np.exp(z[~pos_mask]))
        return result

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        - Use the logistic regression update rule as introduced in lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. 
        - This initialization prevents the weights from starting too large,
        which can cause saturation of the sigmoid function 

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        X_train = self.normalize(X_train, is_train=True)
        N, D = X_train.shape
        self.w = np.random.rand(D)
        for epoch in range(self.epochs):
            z = np.dot(X_train, self.w)
            predictions = self.sigmoid(z)
            error = predictions - y_train
            gradient = np.dot(X_train.T, error) / N
            self.w -= self.lr * gradient
            updated_predictions = self.sigmoid(np.dot(X_train, self.w))
            predicted_labels = (updated_predictions > self.threshold).astype(int)
            accuracy = np.mean(predicted_labels == y_train)
            print(f"Epoch {epoch+1}/{self.epochs}: Training Accuracy = {accuracy*100:.2f}%")
            self.lr *= np.exp(-1 * (epoch * 0.01))


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:exce
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        X_test = self.normalize(X_test, is_train=False)
        probabilities = self.sigmoid(np.dot(X_test, self.w))
        return (probabilities > self.threshold).astype(int)