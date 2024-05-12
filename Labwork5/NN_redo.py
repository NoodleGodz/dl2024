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


def binary_cross_entropy_loss(y_pred, y_true):
    return - (y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))



class Neural_Node:
    def __init__(self) -> None:
        self.a = 0
        self.z = sigmoid(0)

    def update(self,a):
        self.a = a
        self.z = sigmoid(a)

    def raw_update(self,z):
        self.z = z

    def __repr__(self) -> str:
        return f"A: {self.a}, Z: {self.z:.2f}"

class Bias_Node(Neural_Node):
    def __init__(self) -> None:
        self.a = 1
        self.z = 1

    def __repr__(self) -> str:
        return f"Value : {self.z:.2f}"
    
class Layer:
    def __init__(self,numofNeu) -> None:
        self.numofNeu = numofNeu
        self.weight = []
        self.bias_weight = []
        self.neuronlist = []
        for _ in range(numofNeu):
            self.neuronlist.append(Neural_Node())
        self.bias = Bias_Node()

    def random_mode(self,numofNeuNext):
        for _ in range(numofNeuNext):
            self.bias_weight.append(random.random())

        for _ in range(self.numofNeu):
            w = []
            for _ in range(numofNeuNext):
                w.append(random.random())
            
            self.weight.append(w)
        
    def feed_forward(self, inputs,layernum):

        test = []
        if layernum==0:
            for neuron, input in zip(self.neuronlist,inputs):
                neuron.raw_update(input)
                test.append(neuron.z)

        else:
            for neuron, input in zip(self.neuronlist,inputs):
                neuron.update(input)
                test.append(neuron.z)

        #print(inputs)
        if len(inputs)==1:
            return sigmoid(inputs[0])

        outputs = []
        """
        print(f"X =  {test}")
        print(f"Y = {self.weight}")
        print(f"B = {self.bias_weight}")
        """
        outputs = dot_product_with_bias(test,self.weight,self.bias_weight)
        return outputs
    
    def __repr__(self) -> str:
        representation = f"Bias: {self.bias}, Bias Weight: {self.bias_weight}\n"
        for i, neuron in enumerate(self.neuronlist):
            representation += f"Neuron {i+1}: {neuron} Weight {self.weight[i]}\n"
        return representation


    def diff_w(y_pred,y_true):

        return (y_pred - y_true)

    def back_prop(self,y_pred,y_true,lr):
        x = []
        x.append(self.bias.z)

        w = []
        w.append(self.bias_weight)
        for i,j in zip(self.neuronlist,self.weight):
            x.append(i.z)
            w.append(j)
        

        big = []
        for i, weight in zip(x,w):
            empty = []
            for v in weight:
                dw = (y_pred - y_true)*i
                what = v - lr * dw
                empty.append(what)
            big.append(empty)
        
        #update now
        self.bias_weight = big[0]
        self.weight = big[1:]

class Neural_Net:
    def __init__(self, filename) -> None:
        self.nofneural = []
        self.layer = []
        self.random = self.read_textfile(filename=filename)
        for i in range(self.noflayer):
            a = Layer(self.nofneural[i])
            a.random_mode(self.nofneural[i+1])
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
            
    def __repr__(self) -> str:
        representation = ""
        for i, layer in enumerate(self.layer) :
            representation += f"Layer {i}:\n{layer}\n"
        return representation
    
    def feed_forward(self, inputs):
        outputs = inputs
        for i,layer in enumerate(self.layer):
            outputs = layer.feed_forward(outputs,i)
            #print(self)
        return outputs


    def back_prop(self, expected_output, lr ):
        pred_output = self.layer[-1].neuronlist[0].z
        for num in  range(len(self.layer) - 2, -1, -1):
            i = self.layer[num]
            #print(i.numofNeu)
            i.back_prop(pred_output,expected_output,lr)
        return pred_output

    def train_dataset(self, epoches,inputs, expected_output, lr):
        for i in range(epoches):
            losses=[]
            for v,n in zip(inputs,expected_output):
                pred_output =  self.feed_forward(v)
                loss = binary_cross_entropy_loss(pred_output,n)
                losses.append(loss)
                self.back_prop(n,lr)
            print(f"Epoch {i} : Loss {sum(losses)/len(losses)} \n")

    def train(self, epoches,inputs, expected_output, lr):
        for i in range(epoches):

            pred_output =  self.feed_forward(inputs)
            loss = binary_cross_entropy_loss(pred_output,expected_output)
            self.back_prop(expected_output,lr)
            print(f"Epoch {i} : Loss {loss} \n")

random.seed(5)
v = "XOR"
nn = Neural_Net(f"Labwork4\\{v}.txt")
print(nn)
input_data = [1, 0]  # Example input data
output = nn.feed_forward(input_data)
print("Output after forward propagation:", output)
print(nn)
print("\n------------------------------------------\n\n")

learning_rate = 0.02
expected = 1
input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
target_data = [0.001, 1, 1, 0.001]  # XOR

#nn.train_dataset(10000,input_data,expected_output=target_data,lr=learning_rate)

nn.train(2000,input_data[0],expected_output=target_data[0],lr=learning_rate)

print(nn)