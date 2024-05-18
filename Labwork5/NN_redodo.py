import random
import math


def dot_product_with_bias(X, Y, B):
    result = []

    for j in range(len(Y[0])):
        dot = sum(X[i] * Y[i][j] for i in range(len(X)))
        result.append(dot + B[j])

    return result

def dot_product(X, Y):
    result = []

    for j in range(len(Y[0])):
        dot = sum(X[i] * Y[i][j] for i in range(len(X)))
        result.append(dot)

    return result

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def binary_cross_entropy_loss(y_pred, y_true):
    return - (y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))



class Neural_Node:
    def __init__(self) -> None:
        self.a = 0
        self.z = sigmoid(0)
        self.decision = 0

    def update(self,a):
        self.a = a
        self.z = sigmoid(a)
        self.decision = 1 if self.z>0.5 else 0

    def raw_update(self,z):
        self.z = z

    def __repr__(self) -> str:
        return f"A: {self.a}, Z: {self.z:.2f},"

class Bias_Node(Neural_Node):
    def __init__(self) -> None:
        self.a = 1
        self.z = 1

    def __repr__(self) -> str:
        return f"Value : {self.z:.2f}"
    

class NeuralLayer():
    def __init__(self,numofNeu,numofNeuNext) -> None:
        self.numofNeu = numofNeu
        self.numofNeuNext = numofNeuNext
        self.weight = []
        self.bias_weight = []
        self.neuronlist = [Neural_Node() for _ in range(numofNeu)]
        self.bias = Bias_Node()
        self.bias_weight.extend([random.random() for _ in range(numofNeuNext)])
        self.weight.extend([[random.random() for _ in range(numofNeuNext)] for _ in range(self.numofNeu)])

        
    def __repr__(self) -> str:
        representation = f"Bias: {self.bias}, Bias Weight: {self.bias_weight}\n"
        representation += f"Weight: {self.weight}\n"
        for i, neuron in enumerate(self.neuronlist):
            representation += f"Neuron {i+1}: {neuron} \n"
        return representation

    def forward(self, X):
        for i, neuron in enumerate(self.neuronlist):
            neuron.update(X[i])
        Z = [neuron.z for neuron in self.neuronlist]
        activations = dot_product_with_bias(Z, self.weight, self.bias_weight)

        return activations

    def backward(self, delta, learning_rate):
        print("damn",delta)
        error = dot_product(delta,self.weight)
        print(error)
        delta = error * [sigmoid_derivative(i.a) for i in self.neuronlist]
        return delta


class OutputLayer(NeuralLayer):
    def __init__(self, numofNeu):
        self.numofNeu = numofNeu
        self.neuronlist = [Neural_Node() for _ in range(numofNeu)]

    def __repr__(self):
        res = f"Last Layer:\n"
        for i, neuron in enumerate(self.neuronlist, start=1):
            res += f"Neuron {i}: {neuron}\n"
        return res
    
    def forward(self, X):
        for i, neuron in enumerate(self.neuronlist):
            neuron.update(X[i])
        return [neuron.z for neuron in self.neuronlist]

    def backward(self, y, learning_rate=0.1):
        # Compute the output error
        output_error = [self.neuronlist[i].z - y[i] for i in range(self.numofNeu)]

        output_delta = [output_error[i] * sigmoid_derivative(self.neuronlist[i].z) for i in range(self.numofNeu)]
        return output_delta 

class NeuralNetwork:
    def __init__(self, architecture_file):
        with open(architecture_file, 'r') as f:
            lines = f.readlines()
            self.num_layers = int(lines[0])
            self.layer_sizes = [int(line.strip()) for line in lines[1:]]
            self.layer_sizes.append(0)
        self.layers = []
        for i in range(0, self.num_layers -1 ):
            self.layers.append(NeuralLayer(self.layer_sizes[i],self.layer_sizes[i+1]))

        self.layers.append(OutputLayer(1))

    def __repr__(self) -> str:
        representation = ""
        for i, layer in enumerate(self.layers) :
            representation += f"Layer {i}:\n{layer}\n"
        return representation
    
    def forward(self, X):
        output = X 
        for layer in self.layers:
            output = layer.forward(output)
        return output
        
    def backward(self,y,lr = 0.1):
        output = self.layers[-1].neuronlist[0].z
        output_error = output - y
        delta = [output_error * sigmoid_derivative(output)]
        print(output_error)
        print(delta)
        for i in range(self.num_layers-2,-1,-1):
            delta = self.layers[i].backward(delta,lr)
        

random.seed(5)
nn = NeuralNetwork("Labwork5\\XOR.txt")
#print(nn)
X = [0,1]
print(X)
output = (nn.forward(X))

print(nn)


y = 1
nn.backward(y,0.1)