#!/usr/bin/env python
"""
Mode Detection with Deep Neural Network
Implemented in Tensorflow Library(Installing with Anaconda on Windows 10)
The code read the data files from PostgreSQL database
Please find the 'points.csv' and 'labels.csv' on Github and import them into a PostgreSQL db, or
change the code to be able to read all the data from csv files directly.
"""

import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
import matplotlib.pyplot as plt
import numpy as np
from DNN_utils import read_db,segmentation,XY_preparation,convert_to_one_hot, \
    flattening_data,split_train_test, filtering ,normalizing
import time

import pandas as pd

start_time = time.time()


####################Setting  parameters#########################
# number of point per each segment
seg_size = 70
#number of classes and channels
num_classes = 4
num_channels = 4
# number of nodes per each layer ('np


# .repeat' function is used)
# the default is 65 layers (without the output layer)
layer_nods = np.repeat([k for k in range(num_channels*seg_size,0,-5)],2).tolist()
    # [seg_size, 40, 10]
# np.repeat([k for k in range(seg_size,5,-10)],1).tolist()
# number of classes, i.e. number of modes (here: 'walk','bike','car','public transit','car and public transit')


points_table = "segment_trip_id_cleaned_valid_btw_home_work_study"
labels_table = "mode_activity_trip_cleaned_valid_btw_home_work_study"

#################################################################
#X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# layer_nods = [25, 12]
# num_classes = 6
# seg_size = 4096
#
# X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
# X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# print('X_train_orig.shape', X_train_orig.shape)
# # Normalize image vectors
# X_train = X_train_flatten/255.
# X_test = X_test_flatten/255.
# # Convert training and test labels to one hot matrices
# Y_train = convert_to_one_hot(Y_train_orig, 6)
# Y_test = convert_to_one_hot(Y_test_orig, 6)

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar, size of a trip vector (i.e. max number of GPS points along a trip in dataset)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    """

    ### START CODE HERE ### (approx. 2 lines)
    X = tf.placeholder(tf.float32, shape=[n_x, None])
    Y = tf.placeholder(tf.float32, shape=[n_y, None])
    keep_prob = tf.placeholder(tf.float32)

    return X, Y, keep_prob

def initialize_parameters(layer_nods, num_classes, seg_size):
    """
        Initializes parameters to build a neural network with tensorflow. The shapes are:
                            W1 : [25, 3*padding size]
                            b1 : [25, 1]
                            W2 : [12, 25]
                            b2 : [12, 1]
                            W3 : [5, 12]
                            b3 : [5, 1]

        Returns:
        parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
        """
    # tf.set_random_seed(1)  # so that your "random" numbers match ours

    # define the number of nodes for each layer, except the last one
    # add the number of layers for the first and last column
    layer_nods.insert(0, num_channels * seg_size)
    # layer_nods = np.insert(layer_nods,0,3 * seg_size)
    layer_nods.append(num_classes)
    # layer_nods = np.append(layer_nods,num_classes)
    parameters = {}

    for index, current_layer in enumerate(layer_nods):
        if index == 0 or index == len(layer_nods):
            previous_layer = current_layer
            continue
        # declare 'W's
        globals()['W{}'.format(index)] = tf.get_variable('W{}'.format(index),
                                                         ['{}'.format(current_layer), '{}'.format(previous_layer)],
                                                         initializer=tf.contrib.layers.xavier_initializer(seed=1))

        # declare 'b's
        globals()['b{}'.format(index)] = tf.get_variable('b{}'.format(index), ['{}'.format(current_layer), 1],
                                                         initializer=tf.zeros_initializer())

        parameters['W{}'.format(index)] = globals()['W{}'.format(index)]
        parameters['b{}'.format(index)] = globals()['b{}'.format(index)]

        previous_layer = current_layer

    return parameters

def forward_propagation(X, parameters, keep_prob):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3".......
                  the shapes are given in initialize_parameters

    Returns:
    final_Z -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    for index, current_layer in enumerate(layer_nods):
        if index == 0 or index == len(layer_nods):
            previous_layer = current_layer
            continue
        globals()['W{}'.format(index)] = parameters['W{}'.format(index)]
        globals()['b{}'.format(index)] = parameters['b{}'.format(index)]

        if index == len(layer_nods) - 1:
            globals()['Z{}'.format(index)] = \
                tf.add(tf.matmul(globals()['W{}'.format(index)], globals()['A{}'.format(index - 1)]),
                       globals()['b{}'.format(index)])
        else:
            if index == 1:
                # e.g.: Z1 = np.dot(W1, X) + b1
                globals()['Z{}'.format(index)] = \
                    tf.add(tf.matmul(globals()['W{}'.format(index)], X),
                           globals()['b{}'.format(index)])
                # e.g.:# A1 = relu(Z1)
                # globals()['A{}'.format(index)] = tf.nn.tanh(globals()['Z{}'.format(index)])
                #globals()['A{}'.format(index)] = tf.nn.leaky_relu(globals()['Z{}'.format(index)], alpha=0.02)
                #globals()['A{}'.format(index)] = tf.nn.selu(globals()['Z{}'.format(index)])


                #with regularization
                globals()['relu_output{}'.format(index)] = tf.nn.leaky_relu(globals()['Z{}'.format(index)], alpha=0.02)
                globals()['A{}'.format(index)] = tf.nn.dropout(globals()['relu_output{}'.format(index)], keep_prob)



            else:
                # e.g.: Z2 = np.dot(W2, a1) + b2
                globals()['Z{}'.format(index)] = \
                    tf.add(tf.matmul(globals()['W{}'.format(index)],
                                     globals()['A{}'.format(index - 1)]),
                           globals()['b{}'.format(index)])
                # e.g.:# A2 = relu(Z2)
                #globals()['A{}'.format(index)] =
                #  tf.nn.relu(globals()['Z{}'.format(index)])
                #globals()['A{}'.format(index)] = tf.nn.leaky_relu(globals()['Z{}'.format(index)], alpha=0.02)
                #globals()['A{}'.format(index)] = tf.nn.selu(globals()['Z{}'.format(index)])

                # with regularization
                globals()['relu_output{}'.format(index)] = tf.nn.leaky_relu(globals()['Z{}'.format(index)], alpha=0.02)
                globals()['A{}'.format(index)] = tf.nn.dropout(globals()['relu_output{}'.format(index)], keep_prob)


    final_Z = globals()['Z{}'.format(len(layer_nods) - 1)]
    return final_Z


def compute_cost(final_Z, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(final_Z)
    labels = tf.transpose(Y)

    # calculating the cost

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    # Cost function using L2 Regularization
    # regularizer = tf.nn.l2_loss(weights)
    # loss = tf.reduce_mean(loss + beta * regularizer)

    return cost

def model(X_train, Y_train, X_test, Y_test, seg_size, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]  # n_y : output size

    costs = []  # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    X, Y, keep_prob = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters(layer_nods, num_classes, seg_size)

    # Forward propagation: Build the forward propagation in the tensorflow graph
    final_Z = forward_propagation(X, parameters, keep_prob)
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(final_Z, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)

        for epoch in range(num_epochs):

            epoch_cost = 0.  # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                #_, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ##with regularization
                _, minibatch_cost = sess.run([optimizer, cost],
                                             feed_dict={X: minibatch_X, Y: minibatch_Y, keep_prob: 0.84})
                ### END CODE HERE ###

                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        correct_prediction = tf.equal(tf.argmax(final_Z), tf.argmax(Y))


        # print(parameters)
        # print("total cost is:  ", total_cost)
        # # Do the training loop
        #
        # predict_op = tf.argmax(final_Z)
        #
        # predictions, labels_train = sess.run([predict_op, tf.argmax(Y)], feed_dict={X: X_test, Y: Y_test})
        # print(predictions, labels_train)

        # Calculate the correct predictions
        # correct_prediction = tf.equal(tf.argmax(final_Z), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        # print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        ##with regularization
        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train, keep_prob : 1.0}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test, keep_prob : 1.0}))


        confusion = tf.confusion_matrix(labels=tf.argmax(Y), predictions=tf.argmax(final_Z), num_classes=num_classes)
        confusion_mat = confusion.eval({Y: Y_test, X: X_test, keep_prob: 1.0})

        print(confusion_mat)

        # confusion = tf.confusion_matrix(labels=tf.argmax(Y), predictions=predict_op, num_classes=5)
        # confusion_mat = confusion.eval({Y: Y_test, X: X_test})
        #
        # print(confusion_mat)

        return parameters

#parameters = model(X_train, Y_train, X_test, Y_test, seg_size)


def main():
    # points, labels, num_points_per_trip = read_db(points_table, labels_table)
    # points = filtering(points, num_points_per_trip, windows_size = 15)
    #
    # points_segmented = segmentation(points, seg_size, num_points_per_trip)
    # X_orig, Y_orig = XY_preparation(points_segmented, labels, seg_size, num_channels)
    # load data from npy files
    Y_orig = np.load('DNN_labels_data.npy')
    X_orig = np.load('DNN_segmented_data.npy')
    #X_orig = normalizing(X_orig)



    X_flatten = flattening_data(X_orig)
    X_train, X_test, Y_train, Y_test = split_train_test(X_flatten, Y_orig)
    Y_train = convert_to_one_hot(Y_train, 4)
    Y_test = convert_to_one_hot(Y_test, 4)
    parameters = model(X_train, Y_train, X_test, Y_test, seg_size)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()