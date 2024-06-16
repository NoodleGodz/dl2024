import math

from .utils import zeros


def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

def leakyrelu(x, alpha=0.01):
    return x * alpha if x < 0 else x

def leakyrelu_derivative(x, alpha=0.01):
    return alpha if x < 0 else 1

def tanh(x):
    return math.tanh(x)

def tanh_derivative(x):
    return 1 - math.tanh(x) ** 2

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    e_x = [math.exp(i) for i in x]
    sum_e_x = sum(e_x)
    return [i / sum_e_x for i in e_x]


def softmax_derivative(output, d_out):
  jacobian_matrix = [[0.0 for _ in range(len(output))] for _ in range(len(output))]
  for i in range(len(output)):
    for j in range(len(output)):
      jacobian_matrix[i][j] = output[i] * (1 - output[i] if i == j else -output[j])

  d_input = [0.0 for _ in range(len(output))]
  for i in range(len(output)):
    for j in range(len(output)):
      d_input[i] += jacobian_matrix[i][j] * d_out[j]

  return d_input


def bce_loss(predicted, actual):
    predicted = [min(max(p, 1e-12), 1 - 1e-12) for p in predicted]
    losses = [a * math.log(p) + (1 - a) * math.log(1 - p) for p, a in zip(predicted, actual)]
    return -sum(losses) / len(actual)


def bce_loss_derivative(predicted, actual):
    predicted = [min(max(p, 1e-12), 1 - 1e-12) for p in predicted]
    gradients = [(p - a) / (p * (1 - p)) for p, a in zip(predicted, actual)]
    return [g / len(actual) for g in gradients]


def categorical_crossentropy( y_pred, y_true):
    y_pred = [min(max(p, 1e-12), 1 - 1e-12) for p in y_pred]
    loss = -sum(a * math.log(p) for a, p in zip(y_true, y_pred))
    return loss

def categorical_crossentropy_derivative(y_pred, y_true):
    y_pred = [min(max(p, 1e-12), 1 - 1e-12) for p in y_pred]
    gradients = [p - a for p, a in zip(y_pred, y_true)]
    return gradients
