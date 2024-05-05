import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Neural_Node:
    def __init__(self, nout) -> None:
        self.nout = nout
        self.weight = []
        self.value = 0

    def random_mode(self) -> None:
        for _ in range(self.nout):
            self.weight.append(random.random())
    
    def input_mode(self, w):
        self.weight = w
        
    def update(self, x):
        self.value = sigmoid(x)

    def raw_update(self, x):
        self.value = x

    def __repr__(self) -> str:
        return str(self.value)


class Layer:
    def __init__(self, numofNeu, numofNext) -> None:
        self.neuron = numofNeu
        self.next_layer = numofNext
        self.neuronlist = []
        self.bias = None
    
    def random_mode(self):
        self.bias = Neural_Node(1)  # Initialize the bias node
        self.bias.random_mode()      # Randomize the bias weight

        for _ in range(self.neuron):
            a = Neural_Node(self.next_layer)
            a.random_mode()
            self.neuronlist.append(a)

    def input_mode(self, w2):
        self.bias = Neural_Node(1)   # Initialize the bias node
        self.bias.input_mode([w2[0]])  # Set the bias weight
        for w in w2[1:]:
            a = Neural_Node(self.next_layer)
            a.input_mode(w)
            self.neuronlist.append(a)

    def __repr__(self) -> str:
        return str([neuron.value for neuron in self.neuronlist]) + " bias " + str(self.bias.value)

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neuronlist:
            weighted_sum = sum(w * x for w, x in zip(neuron.weight, inputs))
            weighted_sum += self.bias.weight[0]  # Add bias
            neuron.update(weighted_sum)
            outputs.append(neuron.value)
        return outputs

class Neural_Net:
    def __init__(self) -> None:
        self.input_layer = Layer(2, 2)
        self.hidden_layer = Layer(2, 1)
        self.output_layer = Layer(1, 0)

        self.input_layer.random_mode()
        self.hidden_layer.random_mode()
        self.output_layer.random_mode()

    def feed_forward(self, inputs):
        hidden_outputs = self.input_layer.feed_forward(inputs)
        final_outputs = self.hidden_layer.feed_forward(hidden_outputs)
        return self.output_layer.feed_forward(final_outputs)

# Training data for XOR
training_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]

# Training the network
nn = Neural_Net()
learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    total_error = 0
    for input_data, target in training_data:
        output = nn.feed_forward(input_data)

        # Calculate error
        errors = [target[i] - output[i] for i in range(len(target))]
        total_error += sum(errors)

        # Backpropagation
        output_deltas = [error * output[i] * (1 - output[i]) for i, error in enumerate(errors)]
        hidden_deltas = [hidden_layer.weight[i] * output_deltas[0] * (1 - output[i]) for i, output in enumerate(output)]

        # Update weights
        for i, neuron in enumerate(nn.output_layer.neuronlist):
            for j in range(len(neuron.weight)):
                neuron.weight[j] += learning_rate * output_deltas[i] * nn.hidden_layer.neuronlist[j].value

        for i, neuron in enumerate(nn.hidden_layer.neuronlist):
            for j in range(len(neuron.weight)):
                neuron.weight[j] += learning_rate * hidden_deltas[i] * nn.input_layer.neuronlist[j].value

print(sigmoid(-1.5))