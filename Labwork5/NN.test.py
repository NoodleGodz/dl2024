import random
import numpy as np

# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Cross-entropy loss
def cross_entropy_loss(y, y_hat):
    return - (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

# Neural network class
class NeuralNetwork:
    def __init__(self):
        # Initialize weights and biases
        self.input_size = 2
        self.hidden_size_1 = 2
        self.hidden_size_2 = 2
        self.output_size = 1

        self.weights_input_hidden1 = np.random.randn(self.input_size, self.hidden_size_1)
        self.bias_hidden1 = np.zeros((1, self.hidden_size_1))
        
        self.weights_hidden1_hidden2 = np.random.randn(self.hidden_size_1, self.hidden_size_2)
        self.bias_hidden2 = np.zeros((1, self.hidden_size_2))
        
        self.weights_hidden2_output = np.random.randn(self.hidden_size_2, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))
        
        self.learning_rate = 0.2

    def forward(self, X):
        # Forward pass
        self.z1 = np.dot(X, self.weights_input_hidden1) + self.bias_hidden1
        self.a1 = sigmoid(self.z1)
        
        self.z2 = np.dot(self.a1, self.weights_hidden1_hidden2) + self.bias_hidden2
        self.a2 = sigmoid(self.z2)
        
        self.z3 = np.dot(self.a2, self.weights_hidden2_output) + self.bias_output
        self.output = sigmoid(self.z3)
        
        return self.output

    def backward(self, X, y, output):
        # Backward pass
        self.output_error = output - y
        self.output_delta = self.output_error * sigmoid_derivative(output)
        
        self.z2_error = self.output_delta.dot(self.weights_hidden2_output.T)
        self.z2_delta = self.z2_error * sigmoid_derivative(self.a2)
        
        self.z1_error = self.z2_delta.dot(self.weights_hidden1_hidden2.T)
        self.z1_delta = self.z1_error * sigmoid_derivative(self.a1)
        
        # Update weights and biases
        self.weights_hidden2_output -= self.learning_rate * self.a2.T.dot(self.output_delta)
        self.bias_output -= self.learning_rate * np.sum(self.output_delta, axis=0, keepdims=True)
        
        self.weights_hidden1_hidden2 -= self.learning_rate * self.a1.T.dot(self.z2_delta)
        self.bias_hidden2 -= self.learning_rate * np.sum(self.z2_delta, axis=0, keepdims=True)
        
        self.weights_input_hidden1 -= self.learning_rate * X.T.dot(self.z1_delta)
        self.bias_hidden1 -= self.learning_rate * np.sum(self.z1_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 1000 == 0:
                loss = np.mean(cross_entropy_loss(y, output))
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Example usage
if __name__ == "__main__":
    # Training data: XOR problem
    X = np.array([[0, 0]
                  ])
    y = np.array([[0]])

    nn = NeuralNetwork()
    nn.train(X, y)


random.seed(5)
print(nn.output_error.shape)
print(nn.output_delta.shape)
print(nn.z2_error.shape)
print(nn.z2_delta.shape)
print(nn.z1.shape)
print(nn.z2_error)