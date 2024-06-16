from src.Layer import Network, PoolingLayer ,Dense, MultiClassDense
from src.act_func import categorical_crossentropy, categorical_crossentropy_derivative
import random
random.seed(24)

def generate_dataset(num_samples):
    X = []
    y = []
    for _ in range(num_samples):
        x1 = random.randint(0, 1)
        x2 = random.randint(0, 1)
        x3 = random.randint(0, 1)
        x4 = random.randint(0, 1)
        
        # Create a label based on some logic. Here we use a combination of XORs and ANDs.
        if (x1 ^ x2) and (x3 ^ x4):
            output = [1, 0, 0]  # Class 0 in one-hot encoding
        elif (x1 and x3) or (x2 and x4):
            output = [0, 1, 0]  # Class 1 in one-hot encoding
        else:
            output = [0, 0, 1]  # Class 2 in one-hot encoding
        
        X.append([x1, x2, x3, x4])
        y.append(output)
    
    return X, y

if __name__ == "__main__":

    X, y = generate_dataset(25)
    print(y)
    network = Network()
    network.add(Dense(input_dim=4, output_dim=4, activation='relu'))
    network.add(Dense(input_dim=4, output_dim=3, activation='relu'))
    network.add(MultiClassDense(input_dim=3, num_classes=3))

    network.summary(X[0])

    
    network.train(X, y, epochs=6000, learning_rate=0.2, loss_fn=categorical_crossentropy, loss_fn_derivative=categorical_crossentropy_derivative,log_file="log/FCNN_Multi.txt")
    
    #Test the network
    for i, x in enumerate(X) :
        print(f'Input: {x},True {y[i]}, Predicted: {network.predict(x)}')

