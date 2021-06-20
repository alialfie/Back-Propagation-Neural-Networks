import math
import pickle

import numpy

a_file = open("weights.pkl", "rb")
weights = pickle.load(a_file)
print(weights)

a_file = open("dataset.pkl", "rb")
dataset = pickle.load(a_file)
print(dataset)

min = dataset['min']
max = dataset['max']


file_data = {
    'n_inputs': 0,
    'n_hidden': 0,
    'n_outputs': 0,
    'n_data': 0,
    'dataset': []
}

def read_file(file_name, file_data):
    with open(file_name) as f:
        m, l, n = [int(x) for x in next(f).split()]
        k = [int(x) for x in next(f).split()]
        dataset_ = [[float(x) for x in line.split()] for line in f]

        file_data['n_inputs'] = m
        file_data['n_hidden'] = l
        file_data['n_outputs'] = n
        file_data['n_data'] = k[0]
        file_data['dataset'] = dataset_

def activate(weights, inputs):
    x = weights[0]  # because bias = 1 so bias * weight equals weights[0]
    for i in range(len(weights) - 1):
        x += weights[i + 1] * inputs[i]

    # print(x)
    return x

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def get_output(weights, inputs):
    for layer in weights:
        new_inputs = []
        count = 0
        for neuron in weights[layer]:
            output = sigmoid(activate(neuron, inputs))
            new_inputs.append(output)
        inputs = new_inputs
        count += 1
    return inputs

def get_mean(row):
    sum = 0
    for x in row:
        sum += x
    return sum / len(row)


def get_std_dev(row, mean):
    sum = 0
    for x in row:
        sum += (x - mean) ** 2
    return math.sqrt(sum / len(row))


def normalize_inputs(row):
    mean = get_mean(row)
    std_dev = get_std_dev(row, mean)
    for i, x in enumerate(row):
        row[i] = (x - mean) / std_dev
    return row


def denormalize_outputs(predicted, min, max):
    for i in range(len(predicted)):
        predicted[i] = predicted[i] * (max[i] - min[i]) + min[i]
    return predicted

input = normalize_inputs([362.6,  189.0,     0.0,   164.9,    11.6,    944.7,   755.8,     56])

predicted = (get_output(weights, input))
print(denormalize_outputs(predicted, min, max))


def run():
    MSE = 0
    for i in range(file_data['n_data']):
        predicted = get_output(weights, normalize_inputs(file_data['dataset'][i][:-file_data['n_outputs']]))
        expected = file_data['dataset'][i][file_data['n_inputs']:]

        denormalize_outputs(predicted, min, max)

        print(expected, predicted)

        error = 0
        for j in range(file_data['n_outputs']):
            error += (predicted[j] - expected[j]) ** 2
        error = (1 / (2 * file_data['n_outputs'])) * error
        MSE += error
    MSE = MSE / file_data['n_data']
    print(MSE)


read_file("train.txt", file_data)
run()
