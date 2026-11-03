import matplotlib.pyplot as plt
import numpy as np


class Hopfield:
    def __init__(self, shape: tuple[int, int], patterns: list[list[int]]):
        self.shape = shape
        self.size = shape[0] * shape[1]
        self.weights = np.zeros((self.size, self.size))
        self.patterns = patterns

    def hebbian_train(self):
        for pattern in self.patterns:
            self.weights += np.outer(pattern, pattern)
        for i in range(self.size):
            self.weights[i, i] = 0
        self.weights /= len(self.patterns)
        return self.weights

    def normal_initialization(self, variance: float, seed: int = 42):
        generator = np.random.default_rng(seed)
        self.weights = generator.normal(0, variance, (self.size, self.size))

    def oja_train(self, learning_rate: float = 0.01, epochs: int = 400, info_interval: int = 10):
        self.normal_initialization(1 / self.size)
        for epoch in range(1, epochs + 1):
            for pattern in self.patterns:
                out = np.dot(self.weights, pattern)
                column_values = pattern * out
                subtractor = np.stack([column_values] * self.size, axis=0)
                self.weights += learning_rate * (np.outer(pattern, pattern) - subtractor)
            for i in range(self.size):
                self.weights[i, i] = 0
            if epoch % info_interval == 0:
                print(f"Epoch {epoch} completed")

    def recall(self, pattern: list[int]):
        return np.sign(np.dot(self.weights, pattern))

    def recall_until_convergence(self, pattern: list[int]):
        while True:
            new_pattern = self.recall(pattern)
            if np.array_equal(new_pattern, pattern):
                return new_pattern
            pattern = new_pattern

    def plot_pattern(self, pattern: np.ndarray):
        plt.imshow(pattern.reshape(self.shape), cmap="gray")
        plt.show()
