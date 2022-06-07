# https://pywavelets.readthedocs.io/en/latest/#documentation
# https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#custom-wavelets
# https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
# https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html
# https://pywavelets.readthedocs.io/en/latest/ref/nd-dwt-and-idwt.html
# import wavelet transformations
import pywt
# import fourier transformation
import numpy.fft as ft
# import artificial neural network
# https://www.tensorflow.org/
# https://github.com/tensorflow/tensorflow/tree/r1.2
#
import numpy as np
import pickle
import matplotlib.pyplot as plt
# https://matplotlib.org/users/pyplot_tutorial.html
import json
import os
import pandas as pd
# Employing an exorbitant test to find optimal determination algorithms for fingerprinting a human from brain signals.
# __name__ == 'Brian Las Sing'
# Brain Signals
import random
import math
from numpyencoder import NumpyEncoder

# Just use: import tensorflow
class Connection:
    def __init__(self, *args):
        self.weight = 0
        self.randomise_weight()
        self.set_weight(args)

    def randomise_weight(self):
        self.set_weight(random.uniform(0, 2)-1)

    def set_weight(self, temp_weight):
        self.weight = temp_weight


class Neuron:
    def __init__(self, num_of_connections):
        self.bias = None
        self.connections = None
        self.randomise_bias()
        for num_of in range(0, num_of_connections, 1):
            connect = Connection()
            self.add_connection(connect)

    def add_connection(self, conn):
        self.connections.append(conn)

    def get_connection_count(self):
        return len(self.connections)

    def set_bias(self, temp_bias):
        self.bias = temp_bias

    def randomise_bias(self):
        self.set_bias(random.random())

    def get_neuron_output(self, conn_entry_values):
        neuron_input_value = 0
        for con in range(0, self.get_connection_count(), 1):
            neuron_input_value += self.connections[con].calc_conn_exit(conn_entry_values[con])
        neuron_input_value += self.bias
        neuron_output_value = self.activation(neuron_input_value)
        return neuron_output_value

    @staticmethod
    def activation(x):
        activated_value = 1 / (1 + sqrt(1 + (math.exp(-1*x))**2))
        return activated_value


class Layer:
    def __init__(self, number_connections, number_neurons):
        self.layer_error = None
        self.layer_inputs = None
        self.learning_rate = None
        self.expected_outputs = None
        self.actual_outputs = None
        self.neurons = None
        for neu in range(0, number_neurons, 1):
            temp_neuron = Neuron(number_connections)
            self.add_neuron(temp_neuron)
            self.add_actual_output()

    def add_neuron(self, x_neuron):
        self.neurons.append(x_neuron)

    def get_neuron_count(self):
        return len(self.neurons)

    def add_actual_output(self):
        self.actual_outputs = [None for _ in range(0, len(self.actual_outputs)+1)]

    def set_expected_outputs(self, temp_expected_outputs):
        self.expected_outputs = temp_expected_outputs

    def clear_expected_output(self):
        self.expected_outputs = [None for _ in range(0, len(self.expected_outputs))]

    def set_learning_rate(self, temp_learning_rate):
        self.learning_rate = temp_learning_rate

    def set_inputs(self, temp_inputs):
        self.layer_inputs = temp_inputs

    def process_inputs_to_outputs(self):
        for count in range(0, self.get_neuron_count(), 1):
            self.actual_outputs[count] = self.neurons[count].get_neuron_output(self.layer_inputs)

    def get_layer_error(self):
        return self.layer_error

    def set_layer_error(self, temp_layer_error):
        self.layer_error = temp_layer_error

    def increase_layer_error_by(self, temporary_layer_error):
        self.layer_error += temporary_layer_error

    def set_delta_error(self, expected_output_data):
        self.set_expected_outputs(expected_output_data)
        self.set_layer_error(0)
        for n_c in range(0, self.get_neuron_count(), 1):
            self.neurons[n_c].delta_error = self.actual_outputs[n_c] * (1-self.actual_outputs[n_c]) * \
                                            (self.expected_outputs[n_c]-self.actual_outputs[n_c])
            self.increase_layer_error_by(math.fabs(self.expected_outputs[n_c]-self.actual_outputs[n_c]))

    def train_layer(self, temporary_learning_rate):
        self.set_learning_rate(temporary_learning_rate)
        for ne in range(0, self.get_neuron_count(), 1):
            self.neurons[ne].bias += (self.learning_rate * 1 * self.neurons[ne].delta_error)
            for co in range(0, self.neurons[ne].get_connection_count(), 1):
                self.neurons[ne].connections[co].weight += \
                    (self.learning_rate * self.neurons[ne].connections[co].conn_entry * self.neurons[ne].delta_error)


class NeuralNetwork:
    def __init__(self):
        self.learning_rate = 0.1
        self.layers = None
        self.array_of_inputs = None
        self.array_of_outputs = None
        self.network_error = None
        self.data_index = None
        self.training_error = None
        self.training_counter = None
        self.retrain_chances = 0

    def add_layer(self, num_connections, num_neurons):
        self.layers.append(Layer(num_connections, num_neurons))

    def get_layer_count(self):
        return len(self.layers)

    def set_learning_rate(self, temp_learning_rate):
        self.learning_rate = temp_learning_rate

    def set_inputs(self, temp_inputs):
        self.array_of_inputs = temp_inputs

    def set_layer_inputs(self, temporary_inputs, layer_index):
        self.layers[layer_index].set_inputs(temporary_inputs)

    def set_outputs(self, temp_outputs):
        self.array_of_outputs = temp_outputs

    def get_outputs(self):
        return self.array_of_outputs

    def process_inputs_to_outputs(self, tempor_inputs):
        self.set_inputs(tempor_inputs)
        for lc in range(0, self.get_layer_count(), 1):
            if lc == 0:
                self.set_layer_inputs(self.array_of_inputs, lc)
            else:
                self.set_layer_inputs(self.layers[lc-1].actual_outputs, lc)
            self.layers[lc].process_inputs_to_outputs()
        self.set_outputs(self.layers[self.get_layer_count()-1].actual_outputs)

    def train_network(self, input_data, ex_out_data):
        self.process_inputs_to_outputs(input_data)
        for lc in range(self.get_layer_count()-1, -1, -1):      # not sure on reversed range
            if lc == self.get_layer_count()-1:
                self.layers[lc].set_delta_error(ex_out_data)
                self.layers[lc].train_layers(self.learning_rate)
                self.network_error = self.layers[lc].get_layer_error()
            else:
                for nc in range(0, self.layers[lc].get_neuron_count(), 1):
                    self.layers[lc].neurons[nc].delta_error = 0
                    for cc in range(0, self.layers[lc].get_neuron_count(), 1):
                        self.layers[lc].neurons[nc].delta_error += (self.layers[lc+1].neurons[cc].connections[nc].weight
                                                                    * self.layers[lc+1].neurons[cc].delta_error)
                    self.layers[nc].neurons[nc].delta_error *= (self.layers[lc].neurons[nc].neuron_output_value *
                                                                (1-self.layers[lc].neurons[nc].neuron_output_value))
                self.layers[lc].train_layer(self.learning_rate)
                self.layers[lc].clear_expected_output()

    # use array of array for first two

    def training_cycle(self, training_input_data, training_expected_data, train_randomly):
        self.training_error = 0
        for inp in range(0, len(training_input_data), 1):
            if train_randomly:
                self.data_index = random.uniform(0, len(training_input_data))
            else:
                self.data_index = inp
            # .get(self.data_index) used in Processing
            self.train_network(training_input_data[self.data_index], training_expected_data[self.data_index])
            self.training_error += math.fabs(self.network_error)

    def auto_train_network(self, train_input_data, train_expected_data, training_error_target, cycle_limit):
        self.training_error = 9999
        self.training_counter = 0
        while self.training_error > training_error_target and self.training_counter < cycle_limit:
            self.training_error = 0
            self.training_cycle(train_input_data, train_expected_data, True)
            self.training_counter += 1
        if self.training_counter < cycle_limit:
            self.training_cycle(train_input_data, train_expected_data, False)
            self.training_counter += 1
            if self.training_error > training_error_target:
                if self.retrain_chances < 10:
                    self.retrain_chances += 1
                    self.auto_train_network(train_input_data, train_expected_data, training_error_target, cycle_limit)
        else:
            print("Cycle Limit reached. Retrained {} times. Error: {}".format(self.retrain_chances, self.training_error))

def custom_tail(x, a, b):
    dec_lo = [a, a/2]
    dec_hi = [-b, a]
    rec_lo = [a/2, a]
    rec_hi = [a, -b]
    custom = pywt.Wavelet(name="tail", filter_bank=[dec_lo, dec_hi, rec_lo, rec_hi])
    wdax = pywt.dwt(x, custom)
    wiax = custom.wavefun()
    return wdax

def CircularDFT(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.sqrt(1+ (-4j*np.pi*k *np.exp(-4j * np.pi * k * n / N))**2)
    X = np.dot(e, x)
    return X  


def DFT(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-4j * np.pi * k * n / N)
    X = np.dot(e, x)
    return X    



def do_something_special(text, n, j):
    # convert the file into data points,
    # perform all operations of functions on data,
    # re-save the data in a new format

    i = -1
    stats = [[0]*256 for _ in range(0, 64)]
    for tex in text:
        if tex.find('#') > -1:
            if tex.find('FP1') > -1:
                i += 1
                m = -1
            elif tex.find('FP2') > -1:
                i += 1
                m = -1
            elif tex.find('F7') > -1:
                i += 1
                m = -1
            elif tex.find('F8') > -1:
                i += 1
                m = -1
            elif tex.find('AF1') > -1:
                i += 1
                m = -1
            elif tex.find('AF2') > -1:
                i += 1
                m = -1
            elif tex.find('FZ') > -1:
                i += 1
                m = -1
            elif tex.find('F4') > -1:
                i += 1
                m = -1
            elif tex.find('F3') > -1:
                i += 1
                m = -1
            elif tex.find('FC6') > -1:
                i += 1
                m = -1
            elif tex.find('FC5') > -1:
                i += 1
                m = -1
            elif tex.find('FC2') > -1:
                i += 1
                m = -1
            elif tex.find('FC1') > -1:
                i += 1
                m = -1
            elif tex.find('T8') > -1:
                i += 1
                m = -1
            elif tex.find('T7') > -1:
                i += 1
                m = -1
            elif tex.find('CZ') > -1:
                i += 1
                m = -1
            elif tex.find('C3') > -1:
                i += 1
                m = -1
            elif tex.find('C4') > -1:
                i += 1
                m = -1
            elif tex.find('CP5') > -1:
                i += 1
                m = -1
            elif tex.find('CP6') > -1:
                i += 1
                m = -1
            elif tex.find('CP1') > -1:
                i += 1
                m = -1
            elif tex.find('CP2') > -1:
                i += 1
                m = -1
            elif tex.find('P3') > -1:
                i += 1
                m = -1
            elif tex.find('P4') > -1:
                i += 1
                m = -1
            elif tex.find('PZ') > -1:
                i += 1
                m = -1
            elif tex.find('P8') > -1:
                i += 1
                m = -1
            elif tex.find('P7') > -1:
                i += 1
                m = -1
            elif tex.find('PO2') > -1:
                i += 1
                m = -1
            elif tex.find('PO1') > -1:
                i += 1
                m = -1
            elif tex.find('O2') > -1:
                i += 1
                m = -1
            elif tex.find('O1') > -1:
                i += 1
                m = -1
            elif tex.find('X') > -1:
                i += 1
                m = -1
            elif tex.find('AF7') > -1:
                i += 1
                m = -1
            elif tex.find('AF8') > -1:
                i += 1
                m = -1
            elif tex.find('F5') > -1:
                i += 1
                m = -1
            elif tex.find('F6') > -1:
                i += 1
                m = -1
            elif tex.find('FT7') > -1:
                i += 1
                m = -1
            elif tex.find('FT8') > -1:
                i += 1
                m = -1
            elif tex.find('FPZ') > -1:
                i += 1
                m = -1
            elif tex.find('FC4') > -1:
                i += 1
                m = -1
            elif tex.find('FC3') > -1:
                i += 1
                m = -1
            elif tex.find('C6') > -1:
                i += 1
                m = -1
            elif tex.find('C5') > -1:
                i += 1
                m = -1
            elif tex.find('F2') > -1:
                i += 1
                m = -1
            elif tex.find('F1') > -1:
                i += 1
                m = -1
            elif tex.find('TP8') > -1:
                i += 1
                m = -1
            elif tex.find('TP7') > -1:
                i += 1
                m = -1
            elif tex.find('AFZ') > -1:
                i += 1
                m = -1
            elif tex.find('CP3') > -1:
                i += 1
                m = -1
            elif tex.find('CP4') > -1:
                i += 1
                m = -1
            elif tex.find('P5') > -1:
                i += 1
                m = -1
            elif tex.find('P6') > -1:
                i += 1
                m = -1
            elif tex.find('C1') > -1:
                i += 1
                m = -1
            elif tex.find('C2') > -1:
                i += 1
                m = -1
            elif tex.find('PO7') > -1:
                i += 1
                m = -1
            elif tex.find('PO8') > -1:
                i += 1
                m = -1
            elif tex.find('FCZ') > -1:
                i += 1
                m = -1
            elif tex.find('POZ') > -1:
                i += 1
                m = -1
            elif tex.find('OZ') > -1:
                i += 1
                m = -1
            elif tex.find('P2') > -1:
                i += 1
                m = -1
            elif tex.find('P1') > -1:
                i += 1
                m = -1
            elif tex.find('CPZ') > -1:
                i += 1
                m = -1
            elif tex.find('nd') > -1:
                i += 1
                m = -1
            elif tex.find('Y') > -1:
                i += 1
                m = -1
        else:
            m += 1
            te = tex.split()
            stats[i][m] = float(te[3])
    for k in range(0, 64):
        for l in range(4):
            if l == 0:
                statist = custom_tail(stats[k], 0.9, 0.9)
                statistics = statist[0].tolist() + statist[1].tolist()
            if l == 1:
                statistics = stats[k]
            if l == 2:
                statistics = DFT(stats[k])
            if l == 3:
                statistics = CircularDFT(stats[k])
                # in here is where each transformation will go.
                # so if 0, do the first transformation and so on.
                # some transformations (derivatives) have only 255 points rather than 256
                # so ensure that the ANN in the SaveResults section is organised, or use len()
                # Yep your project is as simple as using my code and pumping through a bunch of options
                # And then recording 100 times more data (10kSps) with a few (5) people
                # and seeing if it improves your accuracy beyond 1-2 %
            # statistics = stats[k]
            home = '/Users/adenhandasyde/GitHub/EEG/Transformed Data/'
            actual_file_name = home + to_str(n) + '/' + to_str(j) + '/' + to_str(k) + '/' + to_str(l) + '.json'
            # N is Number person, J is number of test, K is number of channel, L is transformation
            # ANN uses Predicts N from J, comparing different K's and L's over P percent learnt.
            if not os.path.exists(home + to_str(n) + '/'):
                os.makedirs(home + to_str(n) + '/')
            if not os.path.exists(home + to_str(n) + '/' + to_str(j) + '/'):
                os.makedirs(home + to_str(n) + '/' + to_str(j) + '/')
            if not os.path.exists(home + to_str(n) + '/' + to_str(j) + '/' + to_str(k) + '/'):
                os.makedirs(home + to_str(n) + '/' + to_str(j) + '/' + to_str(k) + '/')

            writefile = open(actual_file_name, 'w+')
            writefile.close()
            with open(actual_file_name, 'w') as outfile:
                json.dump(np.abs(statistics), outfile, sort_keys=True, indent=4,
                          ensure_ascii=False, cls=NumpyEncoder)
        # with open ('outfile', 'rb') as fp:
        # stats = pickle.load(fp)
def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))




def to_str(k):
    if k < 10:
        return '00' + str(k)
    elif k < 100:
        return '0' + str(k)
    else:
        return str(k)


def import_prepare_saved():
    # raw data i = , 364 -> 447 2a [84], 337->397 2c [61], 2c1000367, 3a 448->461 [14], 3c0000402
    # 'Raw Data/co2a0000' + 364 + i + '/co2a0000' + 364 + i + '.rd.' + j + '.gz'
    n = 0
    z = 0  # z is missing files
    name = '/Users/adenhandasyde/GitHub/EEG'
    for i in range(0, 84):
        for j in range(0, 120):
            try:
                filename = name + '/Raw Data/co2a0000' + str(364+i) + '/co2a0000' + str(364+i)
                actual_file_name = filename + '.rd.' + to_str(j)
                readfile = open(actual_file_name, 'r')
                lines = readfile.readlines()  # document which files exist to prevent try except
                do_something_special(lines, n, j)
            except FileNotFoundError:
                z += 1
        n += 1
    for i in range(0, 61):
        for j in range(0, 120):
            try:
                filename = name + '/Raw Data/co2c0000' + str(337 + i) + '/co2c0000' + str(337+i)
                actual_file_name = filename + '.rd.' + to_str(j)
                readfile = open(actual_file_name, 'r')
                lines = readfile.readlines()  # document which files exist to prevent try except
                do_something_special(lines, n, j)
            except FileNotFoundError:
                z += 1
        n += 1

    for i in range(0, 14):
        for j in range(0, 120):
            try:
                filename = name + '/Raw Data/co3a0000' + str(448 + i) + '/co3a0000' + str(448+i)
                actual_file_name = filename + '.rd.' + to_str(j)
                readfile = open(actual_file_name, 'r')
                jsonfile = json.load(readfile)  # document which files exist to prevent try except
                do_something_special(lines, n, j)
            except FileNotFoundError:
                z += 1
        n += 1

    for j in range(0, 120):
        try:
            filename = name + '/Raw Data/co2c1000' + str(367) + '/co2c1000' + str(367)
            actual_file_name = filename + '.rd.' + to_str(j)
            readfile = open(actual_file_name, 'r')
            lines = readfile.readlines()  # document which files exist to prevent try except
            do_something_special(lines, n, j)
        except FileNotFoundError:
            z += 1
    n += 1
    for j in range(0, 120):
        try:
            filename = name + '/Raw Data/co3c0000' + str(402) + '/co3c0000' + str(402)
            actual_file_name = filename + '.rd.' + to_str(j)
            readfile = open(actual_file_name, 'r')
            lines = readfile.readlines()  # document which files exist to prevent try except
            do_something_special(lines, n, j)
        except FileNotFoundError:
            z += 1
    n += 1


import_prepare_saved()