"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        np.random.seed(444)
        N, D = X_train.shape
        if self.w is None:
            self.w = np.random.uniform(-0.01, 0.01, size=(D, self.n_class))
        batch_size = 1024
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
                grad = np.zeros_like(self.w)
                for j in range(X_batch.shape[0]):
                    sample = X_batch[j]
                    true_class = y_batch[j]
                    score_true = np.dot(sample, self.w[:, true_class])
                    for class_idx in range(self.n_class):
                        if class_idx == true_class:
                            continue
                        current_score = np.dot(sample, self.w[:, class_idx])
                        if current_score > score_true:
                            grad[:, class_idx] -= sample
                            grad[:, true_class] += sample
                self.w += self.lr * grad
            scores = np.dot(X_train, self.w)
            y_pred = np.argmax(scores, axis=1)
            accuracy = np.mean(y_pred == y_train) * 100
            print(f"Epoch {epoch+1}/{self.epochs}, Accuracy: {accuracy:.2f}%")
            self.lr *= np.exp(-0.1 * epoch)
    def rice_train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        - Use the perceptron update rule as introduced in the Lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        np.random.seed(444)
        num_samples, num_features = X_train.shape
        if self.w is None:
            self.w = np.random.uniform(-0.01, 0.01, size=(num_features, self.n_class))
        for epoch in range(self.epochs):
            indices = np.random.permutation(num_samples)
            for idx in indices:
                sample = X_train[idx]
                true_class = y_train[idx]
                score_true = np.dot(self.w[:,true_class], sample)
                for class_idx in range(self.n_class):
                    current_score = np.dot(self.w[:,class_idx], sample)
                    if class_idx !=y_train[idx] and current_score > score_true:
                        self.w[:,class_idx] -= self.lr * sample
                        self.w[:,true_class] += self.lr * sample
                self.lr /= (1 + 0.001 * epoch)
            preds = np.argmax(X_train.dot(self.w), axis=1)
            acc = np.mean(preds == y_train)
            print("Epoch {} Training Accuracy: {:.2f}%".format(epoch + 1, acc * 100))

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
        scores = np.dot(X_test, self.w)
        predictions = np.argmax(scores, axis=1)
        return predictions