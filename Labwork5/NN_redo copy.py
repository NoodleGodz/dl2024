import math
import random

# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def cross_entropy_loss(y, y_hat):
    return - (y * math.log(y_hat) + (1 - y) * math.log(1 - y_hat))

# Dot product of two vectors
def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

# Initialize weights and biases
def initialize_weights(rows, cols):
    return [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]

def initialize_biases(size):
    return [0 for _ in range(size)]

class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.weights = initialize_weights(input_size, output_size)
        self.bias = initialize_biases(output_size)

    def forward(self, inputs):
        self.inputs = inputs
        self.z = [dot_product(inputs, self.weights[i]) + self.bias[i] for i in range(self.output_size)]
        self.activations = [sigmoid(z) for z in self.z]
        return self.activations

    def backward(self, delta):
        delta_next = [sum(delta[i] * self.weights[i][j] for i in range(self.output_size)) for j in range(self.input_size)]
        delta_prev = [delta_next[j] * sigmoid_derivative(self.activations[j]) for j in range(self.input_size)]

        # Update weights and biases
        for i in range(self.output_size):
            for j in range(self.input_size):
                self.weights[i][j] -= delta_prev[j] * self.inputs[j]
            self.bias[i] -= delta_prev[i]
        
        return delta_next

class NeuralNetwork:
    def __init__(self, filename):
        with open(filename, 'r') as file:
            layer_sizes = [int(line.strip()) for line in file]

        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1]))

        self.learning_rate = 0.1

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output[0]  # Return single value

    def backward(self, X, y, output):
        output_error = output - y
        output_delta = output_error * sigmoid_derivative(output)
        
        delta = [output_delta]
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            for i in range(len(X)):
                output = self.forward(X[i])
                self.backward(X[i], y[i], output)
            if epoch % 1000 == 0:
                loss = sum(cross_entropy_loss(y[i], self.forward(X[i])) for i in range(len(X))) / len(X)
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Example usage
if __name__ == "__main__":
    # Initialize NeuralNetwork from file
    nn = NeuralNetwork("Labwork5\\XOR.txt")

    # Training data: XOR problem
    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    y = [0, 1, 1, 0]

    nn.train(X, y)
