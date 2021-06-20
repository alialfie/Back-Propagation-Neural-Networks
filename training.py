import math
import random
import pickle

import numpy

file_data = {
    'n_inputs': 0,
    'n_hidden': 0,
    'n_outputs': 0,
    'n_data': 0,
    'dataset': []
}

dataset = {
    'inputs': [],
    'outputs': [],
    'min': [],
    'max': []
}


def read_file(file_name, file_data):
    with open(file_name) as f:
        m, l, n = [int(x) for x in next(f).split()]
        k = [int(x) for x in next(f).split()]
        dataset = [[float(x) for x in line.split()] for line in f]

        file_data['n_inputs'] = m
        file_data['n_hidden'] = l
        file_data['n_outputs'] = n
        file_data['n_data'] = k[0]
        file_data['dataset'] = dataset


def get_min_max(dataset):
    mins = []
    maxs = []
    for i in range(file_data['n_outputs']):
        mins.append(9999999999)
        maxs.append(0)

    arr = dataset['outputs']
    for y in arr:
        for i in range(file_data['n_outputs']):

            if mins[i] > y[i]:
                mins[i] = y[i]
            if maxs[i] < y[i]:
                maxs[i] = y[i]

    dataset['min'] = mins
    dataset['max'] = maxs


def mean(arr):
    mean_arr = []

    for row in arr:
        sum = 0
        for x in row:
            sum += x
        mean_arr.append(sum / len(row))

    return mean_arr


def std_dev(arr, mean_arr):
    std_dev_arr = []
    for row in arr:
        sum = 0
        mean = mean_arr[arr.index(row)]
        for x in row:
            sum += (x - mean) ** 2
        std_dev_arr.append(math.sqrt(sum / len(row)))

    return std_dev_arr


def normalize_outputs(arr, min, max):
    for i, row in enumerate(arr):
        for j, x in enumerate(row):
            arr[i][j] = (arr[i][j] - min[j]) / (max[j] - min[j])


def normalize_inputs(arr):
    mean_arr = mean(arr)
    std_dev_arr = std_dev(arr, mean_arr)

    for i, row in enumerate(arr):
        mean_i = mean_arr[i]
        std_dev_i = std_dev_arr[i]
        for j, x in enumerate(row):
            arr[i][j] = (x - mean_i) / std_dev_i


# separates inputs and outputs
def separate_inputs_and_outputs(dataset_file, dataset):
    for row in dataset_file:
        input = row[:-file_data['n_outputs']]
        output = row[file_data['n_inputs']:]
        dataset['inputs'].append(input)
        dataset['outputs'].append(output)


# generate random weights
# dict with two elements, both are 2d arrays
# for example if we have 3 inputs and 2 hidden nodes, the first element in
# the 2d array will be something like [w1, w2, w3] which means the 3 weights
# going into first hidden node from input 1, 2 and 3 respectively
def generate_weights(n_inputs, n_hidden, n_outputs):
    weights = {}
    input_to_hidden = []
    for i in range(n_hidden):
        arr = []
        for j in range(n_inputs + 1):  # plus 1 for the bias
            arr.append(random.uniform(-5, 5))
        input_to_hidden.append(arr)

    weights['hidden'] = input_to_hidden

    hidden_to_output = []
    for i in range(n_outputs):
        arr = []
        for j in range(n_hidden+1):
            arr.append(random.uniform(-5, 5))
        hidden_to_output.append(arr)

    weights['output'] = hidden_to_output

    return weights


# takes the weights and the input values and returns the activation value of the node
# for example if we wanna activate hidden node x, and we have 2 inputs with weights
# w1 and w2, then var weights should be an array that contains w1 and w2 and inputs
# is an array that contains input 1 and input 2
def activate(weights, inputs):
    x = weights[0]  # because bias = 1 so bias * weight equals weights[0]
    for i in range(len(weights) - 1):
        x += weights[i + 1] * inputs[i]

    # print(x)
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# this function takes all the weights and all the inputs and creates the outputs for
# hidden and output nodes
def forward_propagation(weights, row):
    inputs = row
    outputs = {}
    for layer in weights:
        new_inputs = []
        for neuron in weights[layer]:
            output = activate(neuron, inputs)
            new_inputs.append(output)
        outputs[layer] = new_inputs
        inputs = new_inputs
    return outputs


# get the derivative
def get_derivative(output):
    return output * (1.0 - output)


# takes all weights and the outputs from the forward propagation and the expected values
# and returns the deltas
def backward_propagate_error(weights, outputs, expected):
    deltas = {}

    # start with the output neurons
    output_to_hidden_deltas = []
    for i in range(file_data['n_outputs']):
        # print(outputs['output'][i], expected[i],  get_derivative(outputs['output'][i]))
        delta = (expected[i] - outputs['output'][i]) * get_derivative(outputs['output'][i])
        output_to_hidden_deltas.append(delta)
    deltas['output'] = output_to_hidden_deltas

    # then with hidden to input neurons
    hidden_to_input_deltas = []
    weights_from_hidden_to_output = weights['output']
    for i in range(file_data['n_hidden']):
        delta = 0
        for j in range(len(output_to_hidden_deltas)):
            delta += output_to_hidden_deltas[j] * weights_from_hidden_to_output[j][i]

        delta = delta * get_derivative(outputs['hidden'][i])
        hidden_to_input_deltas.append(delta)
    deltas['hidden'] = hidden_to_input_deltas

    return deltas


def update_weights(weights, row, l_rate, outputs, deltas):
    # print(weights)
    for i in range(file_data['n_outputs']):
        for j in range(file_data['n_hidden'] + 1):
            if j == 0:
                weights['output'][i][j] = weights['output'][i][j] + l_rate * deltas['output'][i] * 1
            else:
                weights['output'][i][j] = weights['output'][i][j] + l_rate * deltas['output'][i] * outputs['hidden'][j-1]

    for i in range(file_data['n_hidden']):
        for j in range(file_data['n_inputs'] + 1):  # +1 for bias
            if j == 0:
                weights['hidden'][i][j] = weights['hidden'][i][j] + l_rate * deltas['hidden'][i] * 1  # bias
            else:
                weights['hidden'][i][j] = weights['hidden'][i][j] + l_rate * deltas['hidden'][i] * row[j - 1]
    # print(weights)


# Train a network for a fixed number of epochs
def train_network(weights, dataset, l_rate, n_epoch):
    for epoch in range(n_epoch):
        sum_error = 0
        for i in range(file_data['n_data']):
            outputs = forward_propagation(weights, dataset['inputs'][i])

            error = 0
            for j in range(file_data['n_outputs']):
                error += (dataset['outputs'][i][j] - outputs['output'][j]) ** 2
            error = (1 / (2 * file_data['n_outputs'])) * error
            #error = (1 / file_data['n_outputs']) * error
            sum_error += error

            deltas = backward_propagate_error(weights, outputs, dataset['outputs'][i])
            update_weights(weights, dataset['inputs'][i], l_rate, outputs, deltas)
        sum_error = sum_error / file_data['n_data']
        print('>epoch=%d, lrate=%.3f, error=%.9f' % (epoch, l_rate, sum_error))


def save_weights_to_file(weights):
    a_file = open("weights.pkl", "wb")
    pickle.dump(weights, a_file)
    a_file.close()

def save_dataset_to_file(dataset):
    a_file = open("dataset.pkl", "wb")
    pickle.dump(dataset, a_file)
    a_file.close()


read_file("train.txt", file_data)

separate_inputs_and_outputs(file_data['dataset'], dataset)

normalize_inputs(dataset['inputs'])
get_min_max(dataset)
normalize_outputs(dataset['outputs'], dataset['min'], dataset['max'])

weights = generate_weights(file_data['n_inputs'], file_data['n_hidden'], file_data['n_outputs'])

print(dataset['outputs'])
print(dataset['min'])
print(dataset['max'])

train_network(weights, dataset, 0.9, 100)

save_weights_to_file(weights)
save_dataset_to_file(dataset)