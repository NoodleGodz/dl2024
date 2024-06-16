import random
import math

def random_uniform(low, high):
    return random.uniform(low, high)

def zeros(shape):
    if len(shape) == 1:
        return [0] * shape[0]
    if len(shape) == 2:
        return [[0] * shape[1] for _ in range(shape[0])]
    if len(shape) == 3:
        return [[[0] * shape[2] for _ in range(shape[1])] for _ in range(shape[0])]
    raise ValueError("Unsupported shape dimensions")

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def add(a, b):
    return [x + y for x, y in zip(a, b)]


def scalar_multiply_2d_array(array, scalar):
    result = [[element * scalar for element in row] for row in array]
    return result


def add_2d_arrays(array1, array2):
    if len(array1) != len(array2) or len(array1[0]) != len(array2[0]):
        raise ValueError("Both arrays must have the same dimensions.")

    result = [[array1[i][j] + array2[i][j] for j in range(len(array1[0]))] for i in range(len(array1))]
    return result

def subtract(a, b):
    return [x - y for x, y in zip(a, b)]

def get_shape(input):
    if isinstance(input, list):
        if all(isinstance(i, list) for i in input):
            return (len(input),) + get_shape(input[0])
        else:
            return (len(input),)
    else:
        return ()


def conv2d(input_matrix, filter_matrix, stride=1):
    input_height = len(input_matrix)
    input_width = len(input_matrix[0])
    filter_height = len(filter_matrix)
    filter_width = len(filter_matrix[0])

    output_height = (input_height - filter_height) // stride + 1
    output_width = (input_width - filter_width) // stride + 1

    output_matrix = [[0 for _ in range(output_width)] for _ in range(output_height)]

    for i in range(output_height):
        for j in range(output_width):
            conv_sum = 0
            for m in range(filter_height):
                for n in range(filter_width):
                    conv_sum += input_matrix[i * stride + m][j * stride + n] * filter_matrix[m][n]
            output_matrix[i][j] = conv_sum

    return output_matrix


def conv2d_backward(d_out, input, filter_matrix, stride=1):
    input_height = len(input)
    input_width = len(input[0])
    filter_height = len(filter_matrix)
    filter_width = len(filter_matrix[0])
    output_height = len(d_out)
    output_width = len(d_out[0])
    d_input = [[0 for _ in range(input_width)] for _ in range(input_height)]
    d_filter = [[0 for _ in range(filter_width)] for _ in range(filter_height)]

    for i in range(output_height):
        for j in range(output_width):
            for m in range(filter_height):
                for n in range(filter_width):
                    d_input[i * stride + m][j * stride + n] += d_out[i][j] * filter_matrix[m][n]
                    d_filter[m][n] += d_out[i][j] * input[i * stride + m][j * stride + n]

    return d_input, d_filter

def pad_input(input, padding):
    if padding == 0:
        return input
    n_C, n_H, n_W = get_shape(input)
    padded_input = [[[0 for _ in range(n_W + 2 * padding)] for _ in range(n_H + 2 * padding)] for _ in range(n_C)]
    
    for c in range(n_C):
        for h in range(n_H):
            for w in range(n_W):
                padded_input[c][h + padding][w + padding] = input[c][h][w]
                
    return padded_input