import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def inverse_sigmoid(y):
    return math.log(y / (1 - y))

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
        
    def raw_update(self, x):
        self.value = x

    def __repr__(self) -> str:
        return f"Weight: {self.weight}, Value: {self.value:.2f}"


class Layer:
    def __init__(self, numofNeu, numofNext) -> None:
        self.neuron = numofNeu
        self.next_layer = numofNext
        self.neuronlist = []
        self.bias = 1
    
    def random_mode(self):
        for _ in range(self.neuron - 1):
            a = Neural_Node(self.next_layer)
            a.random_mode()
            self.neuronlist.append(a) 

    def input_mode(self, w2):
        self.w2 = w2
        print(w2)
        self.bias = w2[0]
        for w in w2[1:]:
            a = Neural_Node(self.next_layer)
            a.input_mode(w)
            self.neuronlist.append(a)
            self.finalNode = a

    def __repr__(self) -> str:
        representation = f"Bias: {self.bias}\n"
        for i, neuron in enumerate(self.neuronlist):
            representation += f"Neuron {i+1}: {neuron}\n"
        return representation

    def feed_forward(self, inputs):
        self.update_value(inputs)
        outputs = list(self.bias)
        for j, neuron in enumerate(self.neuronlist) :
            #print(neuron)
            w = neuron.weight
            for i in range(len(outputs)):
                outputs[i] += w[i]*inputs[j]
                
        #print(outputs)
        for i in range(len(outputs)):
            a = sigmoid(outputs[i])
            
            if a>0.5 :
                outputs[i] = 1
            else : 
                outputs[i] = 0


        return outputs
    
    def backprop(self,weight):
        pass

    def update_value(self, inputs):
        if len(inputs)>0:
            for i, neuron in enumerate(self.neuronlist):
                neuron.raw_update(inputs[i])

    def last_layer_mode(self):
        self.bias = [1]
        a = Neural_Node(1)
        a.weight = [1]
        self.neuronlist.append(a)

    

class Neural_Net:
    def __init__(self, filename) -> None:
        self.nofneural = []
        self.layer = {}
        self.random = self.read_textfile(filename=filename)
        for i in range(self.noflayer):
            a = Layer(self.nofneural[i], self.nofneural[i+1])
            if self.random:
                a.random_mode()
            else: 
                if i != self.noflayer -1: 
                    a.input_mode(self.w[i])
                else:
                    a.last_layer_mode()
                self.layer[f"Layer {i}"] = a 

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

    def feed_forward(self, inputs):
        output = inputs
        for i in range(self.noflayer -1):
            output = self.layer[f"Layer {i}"].feed_forward(output)
        self.layer[f"Layer {i+1}"].update_value(output)
        return output
    
    def __repr__(self) -> str:
        representation = ""
        for name, layer in self.layer.items():
            representation += f"{name}:\n{layer}\n"
        return representation
    
v = "XOR"
nn = Neural_Net(f"Labwork4\\{v}.txt")
print(nn.layer)

input_values = [[0, 0], [0, 1], [1, 0], [1, 1]]


print(f"\nTruth Table: {v}")
print("Input 1 | Input 2 | Predicted Output")
print("-------------------------------------")
for input_value in input_values:
    out = nn.feed_forward(input_value)
    #output = 1 if out[0] > 0.5 else 0
    print(f"   {input_value[0]}    |    {input_value[1]}    |        {out[0]}")


print("\n",nn)