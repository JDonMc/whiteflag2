import random
import math


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
        self.actual_outputs = [None for _ in range(0, len(self.actual_outputs)+1), 1]

    def set_expected_outputs(self, temp_expected_outputs):
        self.expected_outputs = temp_expected_outputs

    def clear_expected_output(self):
        self.expected_outputs = [None for _ in range(0, len(self.expected_outputs)), 1]

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
            print "Cycle Limit reached. Retrained {} times. Error: {}" .format(self.retrain_chances,
                                                                               self.training_error)