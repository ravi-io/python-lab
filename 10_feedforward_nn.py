"""
Program 10: Feedforward Neural Network (XOR Problem)
Time Complexity: O(epochs * n * L * w) where n=samples, L=layers, w=neurons
Space Complexity: O(L * w) for weights
"""

import math
import random


def sigmoid(x):
    """Sigmoid activation function."""
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_deriv(output):
    """Derivative of sigmoid given the output."""
    return output * (1.0 - output)


class NeuralNetwork:
    """Feedforward Neural Network with backpropagation."""

    def __init__(self, layer_sizes, lr=0.5):
        """layer_sizes: e.g., [2, 4, 1] for 2 inputs, 4 hidden, 1 output."""
        self.lr = lr
        self.n_layers = len(layer_sizes)
        random.seed(42)

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        for i in range(self.n_layers - 1):
            w = [[random.uniform(-1, 1) for _ in range(layer_sizes[i])]
                 for _ in range(layer_sizes[i + 1])]
            b = [random.uniform(-1, 1) for _ in range(layer_sizes[i + 1])]
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x):
        """Forward pass, returns all layer outputs."""
        outputs = [x]
        for w, b in zip(self.weights, self.biases):
            layer_out = []
            for j in range(len(w)):
                net = sum(w[j][k] * outputs[-1][k]
                          for k in range(len(outputs[-1])))
                layer_out.append(sigmoid(net + b[j]))
            outputs.append(layer_out)
        return outputs

    def train(self, X, y, epochs=5000):
        """Train using backpropagation."""
        for epoch in range(epochs):
            total_error = 0
            for xi, yi in zip(X, y):
                # Forward pass
                outputs = self.forward(xi)

                # Calculate output layer deltas
                target = yi if isinstance(yi, list) else [yi]
                out = outputs[-1]
                deltas = [[] for _ in range(self.n_layers)]
                deltas[-1] = [(target[j] - out[j]) * sigmoid_deriv(out[j])
                              for j in range(len(out))]
                total_error += sum((target[j] - out[j]) ** 2
                                   for j in range(len(out)))

                # Backpropagate deltas
                for l in range(self.n_layers - 2, 0, -1):
                    layer_deltas = []
                    for j in range(len(outputs[l])):
                        err = sum(self.weights[l][k][j] * deltas[l + 1][k]
                                  for k in range(len(deltas[l + 1])))
                        layer_deltas.append(err * sigmoid_deriv(outputs[l][j]))
                    deltas[l] = layer_deltas

                # Update weights and biases
                for l in range(self.n_layers - 1):
                    for j in range(len(self.weights[l])):
                        for k in range(len(self.weights[l][j])):
                            self.weights[l][j][k] += (
                                self.lr * deltas[l + 1][j] * outputs[l][k])
                        self.biases[l][j] += self.lr * deltas[l + 1][j]

            if epoch % 1000 == 0:
                print(f"  Epoch {epoch:5d}, Error: {total_error:.6f}")

    def predict(self, x):
        """Predict output for input x."""
        return self.forward(x)[-1]


# Driver Code
if __name__ == "__main__":
    print("=== Feedforward Neural Network ===")
    print("Solving XOR Problem (not linearly separable)\n")

    # XOR dataset
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0], [1], [1], [0]]

    # Network: 2 inputs -> 4 hidden -> 1 output
    nn = NeuralNetwork([2, 4, 1], lr=0.5)
    nn.train(X, y, epochs=5000)

    print("\nResults:")
    print(f"  {'Input':<10} {'Output':<10} {'Rounded':<10} {'Expected'}")
    for xi, yi in zip(X, y):
        output = nn.predict(xi)[0]
        print(f"  {str(xi):<10} {output:<10.4f} {round(output):<10} {yi[0]}")
