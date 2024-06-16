import pickle
from src.Layer import Dense, Flatten, Network, ConvLayer, PoolingLayer, MultiClassDense
from src.Dataset import load_mnist_dataset, display_images
from src.utils import get_shape
from src.act_func import categorical_crossentropy,categorical_crossentropy_derivative
import random


### U can use the load_fast() tensorflow here just for dataload
train_images, train_labels, test_images, test_labels , val_images, val_labels = load_mnist_dataset(num_samples_per_class=30)
print(f'First training image shape: {get_shape(train_images)}')
print(f'First training image label: {train_labels[0]}')

display_images(train_images,train_labels)
"""
random.seed(155)
network = Network()
network.add(ConvLayer(6,1,16))
network.add(PoolingLayer())
network.add(Flatten())
network.add(Dense(input_dim=1936, output_dim=160, activation='relu'))
network.add(MultiClassDense(160,num_classes=10))

network.summary(train_images[0])
#network.train(train_images, train_labels, epochs=100, learning_rate=0.0003, loss_fn=categorical_crossentropy, loss_fn_derivative=categorical_crossentropy_derivative,val_X=val_images,val_y=val_labels,log_file="CNN_Simple_log.txt")

#network.save_model("CNN_Simple.pkl")

"""
network = Network.load_model("savemodel\CNN_Simple.pkl")

out, test_loss, test_accuracy = network.evaluate( test_images, test_labels,categorical_crossentropy)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
