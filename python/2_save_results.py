import tensorflow as tf
import json
import os
import numpy as np
import pandas as pd
import string
# Need to cycle through below and save the accuracy over P from 1->Max number of recordings per person

home = '/Users/adenhandasyde/GitHub/EEG/'

# actual_file_name = home + to_str(n) + to_str(j) + to_str(k) + to_str(l) + '.pickle'
# N is Number person 0-160, J is number of test 0-120, K is number of channels 0-63, L is transformation 0-10
# ANN uses Predicts N from J, comparing different K's and L's over P percent learnt.
# N max is 121, J is 40, k is 63, l is 11.
person_max = 122
test_max = 30
channel_max = 64
transformation_max = 11


def to_str(i):
    if isinstance(i, int):
        if i < 10:
            return '00' + str(i)
        elif i < 100:
            return '0' + str(i)
        else:
            return str(i)
    else:
        raise ValueError('Is not an int')
# Converts the int into a string of three digits 000-999
# Confirmed error free


def dump_results(layers, channel_number, transformation_number, stats):
    if not os.path.exists(home + 'Results/'):
        os.makedirs(home + 'Results/')
    if not os.path.exists(home + 'Results/' + to_str(layers) + '/'):
        os.makedirs(home + 'Results/' + to_str(layers) + '/')
    if not os.path.exists(home + 'Results/' + to_str(layers) + '/' + to_str(channel_number) + '/'):
        os.makedirs(home + 'Results/' + to_str(layers) + '/' + to_str(channel_number) + '/')
    actual_file_name = home + 'Results/' + to_str(layers) + '/' + to_str(channel_number) + '/' + \
                       to_str(transformation_number) + '.json'
    writefile = open(actual_file_name, 'w+')
    writefile.close()
    with open(actual_file_name, 'w') as fp:
        json.dump(float(stats[0]), fp, sort_keys=True, indent=4, ensure_ascii=False)
# Creates a Results directory if it needs to
# Creates a layers/Channel/Transform
# Saves a JSON in that file with the Stats input
# use to_list() on way through and way back use np.array(json.load(xxx))


def import_inputs_train(inlength, max_people, max_tests, channel_number, transformation_number, proportion):
    inputs = np.zeros([proportion * max_people, inlength])
    counter = 0
    for number_person in range(max_people):
        for test_number in range(0, proportion):
            importing_file = home + 'Transformed Data/' + to_str(number_person) + '/' + to_str(test_number) + '/' \
                             + to_str(channel_number) + '/' + to_str(transformation_number) + '.json'
            with open(importing_file, 'rb') as fp:
                inputs[counter] = json.load(fp)
            counter += 1
    return inputs
# Takes in the initial Proportion of Max People
# Finds the Inputs[test][pers] for a specific Channel and Transformation


def import_inputs_test(inlength, max_persons, max_tests, channel_number, transformation_number, proportion):
    inputs = np.zeros([max_persons * proportion, inlength])
    counter = 0
    for number_person in range(max_persons):
        for test_number in range(proportion, proportion * 2):
            importing_file = home + 'Transformed Data/' + to_str(number_person) + '/' + to_str(test_number) + '/' \
                             + to_str(channel_number) + '/' + to_str(transformation_number) + '.json'

            with open(importing_file, 'rb') as fp:
                inputs[counter] = json.load(fp)
            counter += 1
    return inputs
# Takes in the remaining Proportion of Max people,
# finds the inputs[test][pers] for a specific Channel and Transformation


def generate_train_outputs(max_people, max_tests, proportion, truth_limit):
    total_people = max_people
    train_outputs = np.array([np.zeros(total_people)] * total_people * proportion)
    counter = 0
    for p in range(total_people):
        for tes in range(proportion):
            train_outputs[counter][p] = truth_limit
            counter += 1
    return train_outputs
# The Pth output is only true for the Pth person.
# Needs to be replicated for the number of Tests being tested


def generate_test_outputs(max_people, max_tests, proportion, truth_limit):
    total_people = max_people
    test_outputs = np.array([np.zeros(total_people)] * total_people * proportion)
    counter = 0
    for p in range(total_people):
        for tes in range(proportion, proportion * 2):
            test_outputs[counter][p] = truth_limit
            counter += 1
    return test_outputs


def generate_results(inlength, layers, max_people, max_tests, channel_num, transformation_num, truth_limit):
    steps = int((max_tests-10) / 5)
    this_is_it = [0] * steps
    counting = 0
    for proportion in range(10, 15, 5):
        inputs_test = np.array(import_inputs_test(inlength, max_people, max_tests, channel_num, transformation_num, proportion),
                               dtype=object)
        inputs_train = np.array(import_inputs_train(inlength, max_people, max_tests, channel_num, transformation_num, proportion),
                                dtype=object)

        outputs_train = np.array(generate_train_outputs(max_people, max_tests, proportion, truth_limit))
        outputs_test = np.array(generate_test_outputs(max_people, max_tests, proportion, truth_limit))
        # Training

        x = tf.placeholder(tf.float32, [proportion * max_people, inlength])
        w = tf.Variable(tf.zeros([inlength, layers]))
        # TF.zeros takes in the Shape = [1, 2] => [[0, 0], [0, 0]]
        b = tf.Variable(tf.zeros([layers]))

        # Relativizational:
        # Matrix Multiplier of X and W (nxm) * (mxn) => (n*n)
        y = tf.nn.softmax(tf.matmul(x, w) + b)

        # 1 dimensional Max People long, tried max tests also. tried 122
        y_ = tf.placeholder(tf.float32, [None, layers])

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

        train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

        sess = tf.InteractiveSession()

        tf.global_variables_initializer().run()

        # 100 randoms, 1000 times
        for _ in range(1000):
            batch_xs, batch_ys = inputs_train, outputs_train
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        # eval
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        this_is_it[counting] = sess.run(accuracy, feed_dict={x: inputs_test, y_: outputs_test})
        counting += 1
    return this_is_it

# Returns the accuracy at each proportion for a given layer, transformation and channel
# The max tests and max people are for if we want to change the data sets another 2 dimensions later
# Just as the truth limit can be changed.


def do_one_layer_transformation_channel(layer, transformation, channel, inlength):
    limit = 0.95
    results = generate_results(inlength, layer, person_max, test_max, channel, transformation, limit)
    dump_results(layer, channel, transformation, results)

z = 122  # Z is layer count
t = 10  # T is transformation count
c = 0  # C is channel count
# Each of these need to be maximally explored along the dimension of P, proportion, and A, accuracy
for t in range(0, 4):
    for c in range(64):
        inlength = 256
        do_one_layer_transformation_channel(z, t, c, inlength)