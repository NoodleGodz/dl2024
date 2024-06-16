import pickle
from src.Layer import Dense, Flatten, Network, ConvLayer, PoolingLayer, MultiClassDense
from src.Dataset import load_mnist_dataset, display_images
from src.utils import get_shape
from src.act_func import categorical_crossentropy,categorical_crossentropy_derivative

train_images, train_labels, test_images, test_labels , val_images, val_labels = load_mnist_dataset(num_samples_per_class=20)
print(f'First training image shape: {get_shape(train_images)}')
print(f'First training image label: {train_labels[0]}')



network = Network()
network.add(ConvLayer(4,1,16))
network.add(ConvLayer(6,16,32))
network.add(PoolingLayer(size=2))
network.add(ConvLayer(8,32,64))
network.add(Flatten())
network.add(Dense(input_dim=576, output_dim=320, activation='relu'))
network.add(Dense(input_dim=320, output_dim=128, activation='relu'))
network.add(MultiClassDense(128,num_classes=10))

network.summary(train_images[0])
network.train(train_images, train_labels, epochs=20, learning_rate=0.0001, loss_fn=categorical_crossentropy, loss_fn_derivative=categorical_crossentropy_derivative,val_X=val_images,val_y=val_labels)

with open('model.pkl', 'wb') as f:
    pickle.dump(network, f)



