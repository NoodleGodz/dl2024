import pickle
from .utils import add_2d_arrays, conv2d, conv2d_backward, dot, pad_input, random_uniform,get_shape, scalar_multiply_2d_array,zeros
from . import act_func as af
import random
import time

def __init__():
    pass

class ConvLayer:
    def __init__(self, filter_size, depth , num_filters, stride=1, padding=0):
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_depth = depth
        self.stride = stride
        self.padding = padding
        self.filters = [[[[random_uniform(-1, 1) for _ in range(filter_size)]
                           for _ in range(filter_size)]
                           for _ in range(depth)]  
                           for _ in range(num_filters)]
        self.biases = [random_uniform(-0.1, 0.1) for _ in range(num_filters)]

    def forward(self, input):
        self.input = input
        self.input_padded = pad_input(input, self.padding)
        (n_C_prev, n_H_prev, n_W_prev) = get_shape(self.input_padded)
        n_f = self.num_filters
        stride = self.stride
        output = []
        
        for k in range(n_f):
            conv_sum = None
            for c in range(n_C_prev):
                input_slice = self.input_padded[c]
                this_filter = self.filters[k][c]
                res = conv2d(input_slice,this_filter,stride=stride)
                if conv_sum is None : conv_sum = res
                else : conv_sum = add_2d_arrays(conv_sum,res)
            conv_sum = scalar_multiply_2d_array(conv_sum,self.biases[k]/n_C_prev)
            output.append(conv_sum)
        
        self.output = output
        return output
    
    def backward(self, d_out, learning_rate):
        (n_C_prev, n_H_prev, n_W_prev) = get_shape(self.input_padded)
        n_f = self.num_filters
        stride = self.stride
        
        d_input_padded = [[[0 for _ in range(n_W_prev)] for _ in range(n_H_prev)] for _ in range(n_C_prev)]
        d_filters = [[[[0 for _ in range(self.filter_size)]
                         for _ in range(self.filter_size)]
                         for _ in range(self.num_depth)]  
                         for _ in range(n_f)]
        d_biases = [0 for _ in range(n_f)]

        for k in range(n_f):
            for c in range(n_C_prev):
                input_slice = self.input_padded[c]
                this_filter = self.filters[k][c]
                d_input_slice, d_filter_slice = conv2d_backward(d_out[k], input_slice, this_filter, stride=stride)
                d_input_padded[c] = add_2d_arrays(d_input_padded[c], d_input_slice)
                d_filters[k][c] = add_2d_arrays(d_filters[k][c], d_filter_slice)
            
            d_biases[k] = sum([sum(row) for row in d_out[k]])

        for k in range(n_f):
            for c in range(n_C_prev):
                self.filters[k][c] = add_2d_arrays(self.filters[k][c], scalar_multiply_2d_array(d_filters[k][c], -learning_rate))
            self.biases[k] -= learning_rate * d_biases[k]

        if self.padding > 0:
            d_input = [[row[self.padding:-self.padding] for row in channel[self.padding:-self.padding]] for channel in d_input_padded]
        else:
            d_input = d_input_padded

        return d_input


class PoolingLayer:
    def __init__(self, size=2, stride=1, mode='max'):
        self.size = size
        self.stride = stride
        self.mode = mode
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        if self.stride < self.size: self.stride = self.size
        (n_C_prev, n_H_prev, n_W_prev) = get_shape(input)
        
        n_H = int((n_H_prev - self.size) / self.stride) + 1
        n_W = int((n_W_prev - self.size) / self.stride) + 1
        n_C = n_C_prev
        #print(f"H : {n_H} , {n_W}, {n_C}")
        self.output = [[[0 for _ in range(n_W)] for _ in range(n_H)] for _ in range(n_C)]
        
        for c in range(n_C):
            for h in range(n_H):
                for w in range(n_W):
                    vert_start = h * self.stride
                    vert_end = vert_start + self.size
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + self.size

                    a_slice = [row[horiz_start:horiz_end] for row in input[c][vert_start:vert_end]]
                    #print(get_shape(a_slice))
                    if self.mode == 'max':
                        self.output[c][h][w] = max(max(row) for row in a_slice)
                    elif self.mode == 'avg':
                        self.output[c][h][w] = sum(sum(row) for row in a_slice) / (self.size * self.size)
        
        return self.output

    def backward(self, d_out, learing_rate):
        (n_C_prev, n_H_prev, n_W_prev) = get_shape(self.input)
        d_input = [[[0 for _ in range(n_W_prev)] for _ in range(n_H_prev)] for _ in range(n_C_prev)]
        
        for c in range(len(d_out)):
            for h in range(len(d_out[0])):
                for w in range(len(d_out[0][0])):
                    vert_start = h * self.stride
                    vert_end = vert_start + self.size
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + self.size

                    if self.mode == 'max':
                        a_slice = [row[horiz_start:horiz_end] for row in self.input[c][vert_start:vert_end]]
                        max_value = max(max(row) for row in a_slice)
                        for i in range(self.size):
                            for j in range(self.size):
                                if self.input[c][vert_start + i][horiz_start + j] == max_value:
                                    d_input[c][vert_start + i][horiz_start + j] += d_out[c][h][w]
                    elif self.mode == 'average':
                        gradient = d_out[c][h][w] / (self.size * self.size)
                        for i in range(self.size):
                            for j in range(self.size):
                                d_input[c][vert_start + i][horiz_start + j] += gradient

        return d_input


class Dense:
    def __init__(self, input_dim, output_dim, activation='relu'):
        self.weights = [[random_uniform(-1, 1) for _ in range(output_dim)] for _ in range(input_dim)]
        self.biases = [random_uniform(-1, 1) for _ in range(output_dim)]
        self.activation = activation
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        z = [dot(input, [self.weights[j][i] for j in range(len(input))]) + self.biases[i] for i in range(len(self.biases))]
        self.z = z
        if self.activation == 'relu':
            self.output = [af.relu(x) for x in z]
        elif self.activation == 'sigmoid':
            self.output = [af.sigmoid(x) for x in z]
        elif self.activation == 'tanh':
            self.output = [af.tanh(x) for x in z]
        elif self.activation == 'leakyrelu':
            self.output = [af.leakyrelu(x) for x in z]
        else:
            self.output = z 
        return self.output

    def backward(self, d_out, learning_rate):
        if self.activation == 'relu':
            d_activation = [d_out[i] * af.relu_derivative(self.z[i]) for i in range(len(d_out))]
        elif self.activation == 'sigmoid':
            d_activation = [d_out[i] * af.sigmoid_derivative(self.z[i]) for i in range(len(d_out))]
        elif self.activation == 'tanh':
            d_activation = [d_out[i] * af.tanh_derivative(self.z[i]) for i in range(len(d_out))]
        elif self.activation == 'leakyrelu':
            d_activation = [d_out[i] * af.leakyrelu_derivative(self.z[i]) for i in range(len(d_out))]        
        else:
            d_activation = d_out  # no activation


        # Compute the gradient of the loss with respect to weights and biases
        d_weights = zeros((len(self.input), len(self.biases)))
        d_biases = d_activation.copy()
        d_input = zeros((len(self.input),))
        
        for i in range(len(d_activation)):
            for j in range(len(self.input)):
                d_weights[j][i] = self.input[j] * d_activation[i]
                d_input[j] += self.weights[j][i] * d_activation[i]


        for i in range(len(self.biases)):
            self.biases[i] -= learning_rate * d_biases[i]
            for j in range(len(self.input)):
                self.weights[j][i] -= learning_rate * d_weights[j][i]

        return d_input


class MultiClassDense:
    def __init__(self, input_dim, num_classes):
        self.weights = [[random_uniform(-0.01, 0.01) for _ in range(num_classes)] for _ in range(input_dim)]
        self.biases = [random_uniform(-0.01, 0.01) for _ in range(num_classes)]
        self.input = None
        self.output = None
        self.z = None

    def forward(self, input):
        self.input = input
        z = [dot(input, [self.weights[j][i] for j in range(len(input))]) + self.biases[i] for i in range(len(self.biases))]
        self.z = z
        self.output = af.softmax(z)
        return self.output

    def backward(self, d_out, learning_rate):
        d_activation = af.softmax_derivative(self.output, d_out)

        d_weights = zeros((len(self.input), len(self.biases)))
        d_biases = d_activation.copy()
        d_input = zeros((len(self.input),))
        
        for i in range(len(d_activation)):
            for j in range(len(self.input)):
                d_weights[j][i] = self.input[j] * d_activation[i]
                d_input[j] += self.weights[j][i] * d_activation[i]

        # Update weights and biases
        for i in range(len(self.biases)):
            self.biases[i] -= learning_rate * d_biases[i]
            for j in range(len(self.input)):
                self.weights[j][i] -= learning_rate * d_weights[j][i]

        return d_input


class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            #print("\nBfore forward ",input)
            input = layer.forward(input)
            #print("After forward ",input)
        return input

    def backward(self, d_out, learning_rate):
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out, learning_rate)
        return d_out

    def predict(self, input):
        output = self.forward(input)
        return output

    def train(self, X, y, epochs, learning_rate, loss_fn, loss_fn_derivative, log_file="default_log.txt", val_X=None, val_y=None):
        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = 0
            sqen = list(range(len(X)))
            random.shuffle(sqen)
            
            for i in sqen:
                # Forward pass
                output = self.forward(X[i])
                # Compute loss
                loss = loss_fn(output, y[i])
                total_loss += loss
                
                # Compute gradient of the loss with respect to the output
                d_out = loss_fn_derivative(output, y[i])
                
                # Backward pass
                self.backward(d_out, learning_rate)
                
                # Compute accuracy
                predicted_label = [1 if o >= 0.5 else 0 for o in output]
                if predicted_label == y[i]:
                    correct_predictions += 1

            avg_loss = total_loss / len(X)
            accuracy = correct_predictions / len(X)

            # Validation step
            if val_X is not None and val_y is not None:
                val_total_loss = 0
                val_correct_predictions = 0

                for i in range(len(val_X)):
                    # Forward pass
                    val_output = self.forward(val_X[i])
                    # Compute loss
                    val_loss = loss_fn(val_output, val_y[i])
                    val_total_loss += val_loss
                    
                    # Compute accuracy
                    val_predicted_label = [1 if o >= 0.5 else 0 for o in val_output]
                    if val_predicted_label == val_y[i]:
                        val_correct_predictions += 1

                val_avg_loss = val_total_loss / len(val_X)
                val_accuracy = val_correct_predictions / len(val_X)

                named_tuple = time.localtime() 
                time_string = time.strftime("%H:%M:%S", named_tuple)
                print(f'{time_string} :Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Val Loss: {val_avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
                with open(log_file, 'a') as f:
                    f.write(f'{time_string} :Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Val Loss: {val_avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f} \n')
            else:
                named_tuple = time.localtime() 
                time_string = time.strftime("%H:%M:%S", named_tuple)
                print(f'{time_string} :Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
                with open(log_file, 'a') as f:
                    f.write(f'{time_string} :Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f} \n')


    def summary(self, input):
        print("\n\n")
        print("="*60)
        print("Summary of the network")
        print("="*60)
        print(f"{'Layer name':<20} | {'Input shape':<20} | {'Output shape':<20}")
        print("-"*60)
        
        for layer in self.layers:
            output = layer.forward(input)
            input_shape = get_shape(input)
            output_shape = get_shape(output)
            print(f"{layer.__class__.__name__:<20} | {str(input_shape):<20} | {str(output_shape):<20}")
            input = output

        print("="*60)
        print("The network is runnable...")
        print("\n\n")


    def evaluate(self, test_X, test_y, loss_fn):
        total_loss = 0
        correct_predictions = 0
        output_l = []
        for i in range(len(test_X)):
            output = self.forward(test_X[i])
            loss = loss_fn(test_y[i], output)
            total_loss += loss

            predicted_label = [1 if o >= 0.5 else 0 for o in output]
            if predicted_label == test_y[i]:
                correct_predictions += 1

            output_l.append(predicted_label)
        avg_loss = total_loss / len(test_X)
        accuracy = correct_predictions / len(test_X)

        return output_l, avg_loss, accuracy

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, input):
        self.input_shape = get_shape(input)
        flattened_output = [element for sublist in input for subsublist in sublist for element in subsublist]
        return flattened_output

    def backward(self, d_out, learing_rate):
        d_input = zeros(self.input_shape)
        idx = 0
        for i in range(self.input_shape[0]):
            for j in range(self.input_shape[1]):
                for k in range(self.input_shape[2]):
                    d_input[i][j][k] = d_out[idx]
                    idx += 1
        return d_input
