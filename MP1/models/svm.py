"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        N, D = X_train.shape
        scores = X_train.dot(self.w)
        correct_class_scores = scores[np.arange(N), y_train].reshape(N, 1)
        margins = scores - correct_class_scores + 1  
        margins[np.arange(N), y_train] = 0  
        mask = (margins > 0).astype(float) 
        count_positive = np.sum(mask, axis=1)  
        mask[np.arange(N), y_train] = -count_positive
        dW_data = (X_train.T).dot(mask) / N
        dW = self.reg_const * self.w + dW_data
        return dW

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        np.random.seed(444)
        N, D = X_train.shape
        self.w = np.random.uniform(-0.1, 0.1, size=(D, self.n_class))
        batch_size = 200
        num_batches = int(np.ceil(N / batch_size))
        for epoch in range(self.epochs):
            indices = np.arange(N)
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            for i in range(num_batches):
                start = i * batch_size
                end = min(start + batch_size, N)
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]
                grad = self.calc_gradient(X_batch, y_batch)
                self.w -= self.lr * grad
            scores = np.dot(X_train, self.w)
            y_pred = np.argmax(scores, axis=1)
            accuracy = np.mean(y_pred == y_train) * 100
            print(f"Epoch {epoch+1}/{self.epochs}, Accuracy: {accuracy:.2f}%")
            self.lr *= np.exp(-0.1 * epoch)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        scores = X_test.dot(self.w)  # shape: (N, n_class)
        predictions = np.argmax(scores, axis=1)
        return predictions
