import random
import math

def dot_product_with_bias(X, Y, B):
    result = []

    for j in range(len(Y[0])):
        dot = sum(X[i] * Y[i][j] for i in range(len(X)))
        result.append(dot + B[j])

    return result

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Neural_Node:
    def __init__(self):
        self.a = 0
        self.z = sigmoid(0)

    def update(self, a):
        self.a = a
        self.z = sigmoid(a)

    def raw_update(self, z):
        self.z = z

    def __repr__(self):
        return f"A: {self.a}, Z: {self.z:.2f}"

class Bias_Node(Neural_Node):
    def __init__(self):
        self.a = 1
        self.z = 1

    def __repr__(self):
        return f"Value : {self.z:.2f}"

class Layer:
    def __init__(self, numofNeu):
        self.numofNeu = numofNeu
        self.weight = []
        self.bias_weight = []
        self.neuronlist = []
        for _ in range(numofNeu):
            self.neuronlist.append(Neural_Node())
        self.bias = Bias_Node()

    def random_mode(self, numofNeuNext):
        for _ in range(numofNeuNext):
            self.bias_weight.append(random.random())

        for _ in range(self.numofNeu):
            w = []
            for _ in range(numofNeuNext):
                w.append(random.random())

            self.weight.append(w)

    def feed_forward(self, inputs):
        test = []
        for neuron, input in zip(self.neuronlist, inputs):
            neuron.raw_update(input)
            test.append(neuron.z)

        outputs = dot_product_with_bias(test, self.weight, self.bias_weight)
        return outputs

    def __repr__(self):
        representation = f"Bias: {self.bias}, Bias Weight: {self.bias_weight}\n"
        for i, neuron in enumerate(self.neuronlist):
            representation += f"Neuron {i + 1}: {neuron} Weight {self.weight[i]}\n"
        return representation

class Neural_Net:
    def __init__(self, filename):
        self.nofneural = []
        self.layer = []
        self.random = self.read_textfile(filename=filename)
        for i in range(self.noflayer):
            a = Layer(self.nofneural[i])
            a.random_mode(self.nofneural[i + 1])
            self.layer.append(a)

    def read_textfile(self, filename):
        with open(file=filename, mode="r+") as f:
            self.noflayer = int(f.readline())
            for _ in range(self.noflayer):
                self.nofneural.append(int(f.readline()))

            self.nofneural.append(0)
            w = eval(f.readline())
            if not w:
                return True
            else:
                self.w = w
                return False

    def __repr__(self):
        representation = ""
        for i, layer in enumerate(self.layer):
            representation += f"Layer {i}:\n{layer}\n"
        return representation

    def feed_forward(self, inputs):
        outputs = inputs
        for layer in self.layer:
            outputs = layer.feed_forward(outputs)
        return outputs

    def back_propagation(self, inputs, target, learning_rate):
        # Forward pass
        outputs = self.feed_forward(inputs)

        # Backward pass
        delta_outputs = [(o - t) * sigmoid_derivative(o) for o, t in zip(outputs, target)]

        for i in reversed(range(len(self.layer))):
            layer = self.layer[i]

            # Calculate delta for weights
            delta_weights = [[delta_outputs[j] * sigmoid(layer.neuronlist[k].a) * learning_rate
                              for j in range(len(delta_outputs))]
                             for k in range(len(layer.neuronlist))]

            # Update weights
            for j in range(len(layer.weight)):
                for k in range(len(layer.weight[j])):
                    layer.weight[j][k] -= delta_weights[k][j]

            # Calculate delta for biases
            delta_bias_weights = [delta_outputs[j] * learning_rate
                                  for j in range(len(delta_outputs))]

            # Update bias weights
            for j in range(len(layer.bias_weight)):
                layer.bias_weight[j] -= delta_bias_weights[j]

            # Calculate delta for next layer
            if i > 0:
                delta_outputs = [sum(delta_outputs[j] * layer.weight[k][j]
                                     for j in range(len(delta_outputs)))
                                 for k in range(len(layer.neuronlist))]

    def train(self, inputs, targets, learning_rate, epochs):
        for epoch in range(epochs):
            for input_data, target in zip(inputs, targets):
                self.back_propagation(input_data, target, learning_rate)

# Example usage
random.seed(5)
v = "XOR"
nn = Neural_Net(f"Labwork4\\{v}.txt")

# Example input and target data
input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
target_data = [[0], [1], [1], [0]]  # XOR

# Train the neural network
nn.train(input_data, target_data, learning_rate=0.1, epochs=1000)

# Test the trained neural network
for i, input_data in enumerate(input_data):
    output = nn.feed_forward(input_data)
    print(f"Input: {input_data}, Target: {target_data[i]}, Output: {output}")
