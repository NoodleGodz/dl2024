from src.Layer import Dense, Flatten, Network, ConvLayer, PoolingLayer, MultiClassDense
from src.Dataset import load_mnist_dataset, display_images, load_catdog_dataset, load_fast
from src.utils import get_shape
from src.act_func import bce_loss, categorical_crossentropy,categorical_crossentropy_derivative, bce_loss_derivative
import random


### For this file to work, pls download the dataset from "https://www.kaggle.com/datasets/tongpython/cat-and-dog" and extract here, or use the load_cat_dog_fast() tensorflow here just for dataload
#train_images, train_labels, test_images, test_labels , val_images, val_labels = load_catdog_dataset(num_samples_per_class=70)
train_images, train_labels, test_images, test_labels , val_images, val_labels = load_fast("savemodel/catdog_dataset.pkl")
print(f'First training image shape: {get_shape(train_images)}')
print(f'First training image label: {train_labels[0]}')

display_images(train_images,train_labels)

"""
random.seed(155)
network = Network()
network.add(ConvLayer(10,3,16,stride=2))
network.add(PoolingLayer())
network.add(Flatten())
network.add(Dense(input_dim=1600, output_dim=160, activation='relu'))
network.add(Dense(input_dim=160, output_dim=1, activation='sigmoid'))

network.summary(train_images[0])
network.train(train_images, train_labels, epochs=50, learning_rate=0.0003, loss_fn=bce_loss, loss_fn_derivative=bce_loss_derivative,val_X=val_images,val_y=val_labels,log_file="CNN_Binary_log.txt")

network.save_model("CNN_Binary.pkl")
"""

network = Network.load_model("savemodel\CNN_Binary.pkl")
network.summary(train_images[0])
#display_images(test_images,test_labels)
output, test_loss, test_accuracy = network.evaluate( test_images, test_labels,bce_loss)



from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

predictions = output

print("Classification Report:")
print(classification_report(test_labels, predictions))


cm = confusion_matrix(test_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False,
            xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
