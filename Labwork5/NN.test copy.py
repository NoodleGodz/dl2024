import math
import random

# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Cross-entropy loss
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

# Neural network class
class NeuralNetwork:
    def __init__(self):
        # Initialize weights and biases
        self.input_size = 2
        self.hidden_size_1 = 2
        self.hidden_size_2 = 2
        self.output_size = 1

        self.weights_input_hidden1 = initialize_weights(self.input_size, self.hidden_size_1)
        self.bias_hidden1 = initialize_biases(self.hidden_size_1)
        
        self.weights_hidden1_hidden2 = initialize_weights(self.hidden_size_1, self.hidden_size_2)
        self.bias_hidden2 = initialize_biases(self.hidden_size_2)
        
        self.weights_hidden2_output = initialize_weights(self.hidden_size_2, self.output_size)
        self.bias_output = initialize_biases(self.output_size)
        
        self.learning_rate = 0.2

    def forward(self, X):
        # Forward pass
        self.z1 = [dot_product(X, self.weights_input_hidden1[i]) + self.bias_hidden1[i] for i in range(self.hidden_size_1)]
        self.a1 = [sigmoid(z) for z in self.z1]
        
        self.z2 = [dot_product(self.a1, self.weights_hidden1_hidden2[i]) + self.bias_hidden2[i] for i in range(self.hidden_size_2)]
        self.a2 = [sigmoid(z) for z in self.z2]
        
        self.z3 = dot_product(self.a2, [self.weights_hidden2_output[i][0] for i in range(self.hidden_size_2)]) + self.bias_output[0]
        self.output = sigmoid(self.z3)
        
        return self.output

    def backward(self, X, y, output):
        # Backward pass
        output_error = output - y
        output_delta = output_error * sigmoid_derivative(output)

        z2_error = [output_delta * self.weights_hidden2_output[i][0] for i in range(self.hidden_size_2)]
        z2_delta = [z2_error[i] * sigmoid_derivative(self.a2[i]) for i in range(self.hidden_size_2)]
        
        z1_error = [sum(z2_delta[j] * self.weights_hidden1_hidden2[j][i] for j in range(self.hidden_size_2)) for i in range(self.hidden_size_1)]
        z1_delta = [z1_error[i] * sigmoid_derivative(self.a1[i]) for i in range(self.hidden_size_1)]
        
        # Update weights and biases
        for i in range(self.hidden_size_2):
            self.weights_hidden2_output[i][0] -= self.learning_rate * output_delta * self.a2[i]
        self.bias_output[0] -= self.learning_rate * output_delta
        
        for i in range(self.hidden_size_2):
            for j in range(self.hidden_size_1):
                self.weights_hidden1_hidden2[i][j] -= self.learning_rate * z2_delta[i] * self.a1[j]
            self.bias_hidden2[i] -= self.learning_rate * z2_delta[i]
        
        for i in range(self.hidden_size_1):
            for j in range(self.input_size):
                self.weights_input_hidden1[i][j] -= self.learning_rate * z1_delta[i] * X[j]
            self.bias_hidden1[i] -= self.learning_rate * z1_delta[i]

    def train(self, X, y, epochs=100000):
        for epoch in range(epochs):
            for i in range(len(X)):
                output = self.forward(X[i])
                self.backward(X[i], y[i], output)
            if epoch % 1000 == 0:
                loss = sum(cross_entropy_loss(y[i], self.forward(X[i])) for i in range(len(X))) / len(X)
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Example usage
if __name__ == "__main__":
    # Training data: XOR problem
    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    y = [0, 1, 1, 0]

    nn = NeuralNetwork()
    nn.train(X, y)
