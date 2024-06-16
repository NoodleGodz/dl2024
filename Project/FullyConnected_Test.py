from src.Layer import  Dense,Network
from src.act_func import bce_loss, bce_loss_derivative
import random
random.seed(324)


def generate_dataset(num_samples):
    X = []
    y = []
    for _ in range(num_samples):
        x1 = random.randint(0, 1)
        x2 = random.randint(0, 1)
        x3 = random.randint(0, 1)
        output = 1 if x1 ^ x2 ^ x3 else 0 
        X.append([x1, x2, x3])
        y.append([output])
    return X, y



if __name__ == "__main__":

    X, y = generate_dataset(15)
    network = Network()
    network.add(Dense(input_dim=3, output_dim=4, activation='relu'))
    network.add(Dense(input_dim=4, output_dim=3, activation='relu'))
    network.add(Dense(input_dim=3, output_dim=1, activation='sigmoid'))

    network.train(X, y, epochs=4000, learning_rate=0.01, loss_fn=bce_loss, loss_fn_derivative=bce_loss_derivative,log_file="FCNN_Binary.txt")
    network.summary(X[0])

    for i, x in enumerate(X) :
        print(f'Input: {x},True {y[i]}, Predicted: {network.predict(x)}')

