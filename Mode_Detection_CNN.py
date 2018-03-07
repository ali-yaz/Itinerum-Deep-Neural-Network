#!/usr/bin/env python
"""
Mode Detection with Deep Neural Network
Implemented in Tensorflow Library(Installing with Anaconda on Windows 10)
The code read the data files from PostgreSQL database
Please find the 'points.csv' and 'labels.csv' on Github and import them into a PostgreSQL db, or
change the code to be able to read all the data from csv files directly.
"""

import psycopg2
import numpy as np
import time
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
import math
from sklearn.model_selection import train_test_split

# import matplotlib.pyplot as plt
# from sklearn.model_selection import KFold
# ==============================================================================
__author__ = "Ali Yazdizadeh"
__date__ = "February 2018"
__email__ = "ali.yazdizadeh@mail.concordia.ca"
__python_version__ = "3.5.4"
# ==============================================================================

start_time = time.time()

####################Setting  parameters#########################
# number of point per each segment
seg_size = 16*16
n_H = n_W = int(np.sqrt(seg_size))
# number of channels
num_channels = 3
# Define weight parameters' structure to build convolutional neural network
##number of conv layers?
num_layers = 10
##size of each filter (i.e. number of pixels per width or height)
filters_size = np.repeat([k for k in range(4,0,-2)],5).tolist()
##number of filters for each layer
num_filters = np.repeat([k for k in range(96,384,70)],2).tolist()
#define the stride size for each CONV2D layer
num_stride_conv2d = [1 for k in range(0,10)]
#define the stride size for each MAXPOOL layer
num_stride_maxpool = np.tile([k for k in range(4,0,-2)],5).tolist()
##Define the weights
weights = list()
for index, f in enumerate(filters_size):
    if index == 0:
        weights.append([f,f,num_channels,num_filters[index]])
    else:
        weights.append([f, f, num_filters[index - 1], num_filters[index]])

points_table = "segment_trip_id_cleaned_valid_btw_home_work_study"
labels_table = "mode_activity_trip_cleaned_valid_btw_home_work_study"

#layer_nods = np.repeat([k for k in range(seg_size, 5, -5)], 5).tolist()
# number of classes, i.e. number of modes (here: 'walk','bike','car','public transit','car and public transit')
num_classes = 5


#############Make connection to the PostgreSQL database##########
db_conn = psycopg2.connect(user='postgres', password='postgresql', host='localhost', port=5432, database='MtlTrajet_tout_July2017')
db_conn.autocommit = True
db_cur = db_conn.cursor()
# ####################Read the point and labels data#########################
def read_db(points_table, labels_table):
    """

    :param point_table:
    :param labels_table:
    :return: pandas dataframes 'points', 'labels'
    """
    query = """SELECT * FROM {} where (uuid = '001DCAB0-2E98-42E2-85EB-CF297A3534EF'
    	and trip_id = 10) or (uuid = '001DCAB0-2E98-42E2-85EB-CF297A3534EF'
    	and trip_id = 19)

    """
    """
    where uuid in
    (select distinct(uuid) from {} order by uuid)
    order by uuid, trip_id, timestamp
    ;"""
    #'points','com_names' in follwoing commands are panda dataframe type
    points = pd.read_sql_query(query.format(points_table),con=db_conn)



    #read the label data
    query = "SELECT uid,trip_id,mode FROM {};"
    labels = pd.read_sql_query(query.format(labels_table),con=db_conn)



    num_points_per_trip = points.groupby(['uuid', 'trip_id']).size().reset_index(name='counts')
    print('total numnber of points is:',num_points_per_trip['counts'].sum())

    return points, labels, num_points_per_trip



#############Reading Data from CSV##########
# points = pd.read_csv('D:/OneDrive - Concordia University - Canada/Concordia/Thesis/data/mode_detection_January_2018_DNN/points.csv',
#                      names=['uuid', 'trip_id', 'time_interval', 'distance_prev_point', 'speed'])
#
# # points = pd.read_csv('C:/Users/al_ya/PycharmProjects/Itinerum-Deep-Neural-Network/points.csv',
# #                      names=['uuid', 'trip_id', 'time_interval', 'distance_prev_point', 'speed'])
# print(points.shape)
# #print(points.head)
#
#
# labels = pd.read_csv('C:/Users/al_ya/PycharmProjects/Itinerum-Deep-Neural-Network/labels.csv',
#                      names=['uuid', 'trip_id', 'mode'])

# Calculate the number of points along each trip in pandas dataframe
# num_points_per_trip = points.groupby(['uuid', 'trip_id']).size().reset_index(name='counts')

#points = points.iloc[:640]
# #############Make connection to the PostgreSQL database##########
# db_conn = psycopg2.connect(user='postgres', password='postgresql', host='localhost', port=5432,
#                            database='MtlTrajet_tout_July2017')
# db_conn.autocommit = True
# db_cur = db_conn.cursor()
#
# ####################Read the point and labels data#########################
# points_table = "points"
# query = """SELECT * FROM {}
# where uuid in
# (select distinct(uuid) from {} order by uuid )
# order by uuid, trip_id, timestamp
# ;"""
# # 'points','com_names' in follwoing commands are panda dataframe type
# points = pd.read_sql_query(query.format(points_table, points_table), con=db_conn)
#
# # read the column names of points table
# query = """
# select column_name from information_schema.columns where table_name = '{}';
# """
# col_names = pd.read_sql_query(query.format(points_table), con=db_conn)
#
# # read the label data
# labels_table = "labels"
# query = "SELECT uid,trip_id,mode FROM {};"
# labels = pd.read_sql_query(query.format(labels_table), con=db_conn)
#
# # Calculate the number of points along each trip in pandas dataframe
# num_points_per_trip = points.groupby(['uuid', 'trip_id']).size().reset_index(name='counts')


#############create new dataframe with fixed size segments##########
def segmentation(points, seg_size,num_points_per_trip):
    i = 0
    points_segmented = pd.DataFrame()
    # give same size for each segment
    for index, row in num_points_per_trip.iterrows():
        # print(row[2])
        # if row[2] < 2:
        #     continue
        segment_counter = 0
        trip = points.loc[(points['uuid'] == row[0]) & (points['trip_id'] == row[1])]
        #print('trip is:', trip)
        num_segs = math.ceil(row[2] / seg_size)
        print('num_segs is:',num_segs)

        #  loop for splitting the points of a trip into separate 'seg_size' segments.
        for j in range(1, num_segs + 1):


            segment_counter += 1
            b_loc = (j - 1) * seg_size
            e_loc = (j - 1) * seg_size + (seg_size)
            print(b_loc, e_loc)
            print('locations are printed above')
            if j == num_segs and row[2] % seg_size != 0:
                # print('j in the second loop is:', j)
                # print('the first part is processed')
                e_loc = row[2]
                trip = trip.assign(segment_id=segment_counter)
                temp = pd.DataFrame(0, index=np.arange(seg_size), columns=list(trip.columns.values))
                temp.iloc[0:row[2] % seg_size] = trip.iloc[b_loc:e_loc]
                # print('temp is printed and the shape is', temp.shape)
                # print('index of trip is',index)
                points_segmented = points_segmented.append(temp, ignore_index=False)
                continue
            trip = trip.assign(segment_id=segment_counter)
            temp = pd.DataFrame(0, index=np.arange(seg_size), columns=list(trip.columns.values))
            temp.iloc[0:row[2] % seg_size] = trip.iloc[b_loc:e_loc]
            # print('j in the first loop is:', j)
            # print('temp is printed and the shape is', temp.shape)
            # print('index of trip is', index)
            # time.sleep(10)
            points_segmented = points_segmented.append(temp, ignore_index=False)
            points_segmented = points_segmented.append(trip.iloc[b_loc:e_loc], ignore_index=False)
    # drop trips with na or zero values in 'uuid','trip_id','segment_id'
    points_segmented = points_segmented.dropna(subset=['uuid', 'trip_id', 'segment_id'])
    points_segmented = points_segmented[(points_segmented['uuid'] != 0) &
                                        (points_segmented['trip_id'] != 0) &
                                        (points_segmented['segment_id'] != 0)]

    return points_segmented


#############Preparing the X and Y data to feed to neural net##########
def XY_preparation(points_segmented, labels, seg_size, num_channels):
    # Flatten the training and test sets
    num_segements = \
    points_segmented.drop_duplicates(subset=('uuid', 'trip_id', 'segment_id'), keep='first', inplace=False).shape[0]
    print('num segments', num_segements)

    uuid_trip_id_segments = \
    points_segmented.drop_duplicates(subset=('uuid', 'trip_id', 'segment_id'), keep='first', inplace=False)[
        ['uuid', 'trip_id', 'segment_id']]

    print('uuid_trip_id_segments shape is')
    print(uuid_trip_id_segments.shape)


    # creating X_orig and Y_orig arrays
    X_orig = np.zeros((num_segements, n_H, n_W, num_channels))
    X_orig[X_orig == 0] = 0
    Y_orig = np.zeros(num_segements, dtype=int)
    # number of points along each trip
    # print(uuid_trip_id_segments.shape)
    i = 0
    # assign the label for each trip
    for index, row in uuid_trip_id_segments.iterrows():
        # select all the points for each segment
        trip = points_segmented.loc[(points_segmented['uuid'] == row[0]) &
                                    (points_segmented['trip_id'] == row[1]) &
                                    (points_segmented['segment_id'] == row[2])]
        print

        #create numpy arfor channels
        Channel_1 = np.zeros((seg_size))
        Channel_2 = np.zeros((seg_size))
        Channel_3 = np.zeros((seg_size))



        Channel_1[0:trip.shape[0]] = trip.loc[:, 'time_interval']
        Channel_2[0:trip.shape[0]] = trip.loc[:, 'distance_prev_point']
        Channel_3[0:trip.shape[0]] = trip.loc[:, 'speed']


        # Channel_1.append(pd.Series([0 for i in range(0, seg_size - trip.shape[0])]))
        # Channel_2.append(pd.Series([0 for i in range(0, seg_size - trip.shape[0])]))
        # Channel_3.append(pd.Series([0 for i in range(0, seg_size - trip.shape[0])]))

        #
        # Channel_1 = tf.pad(Channel_1, (1, n_H - trip.shape[0]), "CONSTANT")
        # Channel_2 = tf.pad(Channel_2, (1, n_H - trip.shape[0]), "CONSTANT")
        # Channel_3 = tf.pad(Channel_3, (1, n_H - trip.shape[0]), "CONSTANT")


        # aasing the labels to each segment
        label = labels.loc[(labels['uid'] == row[0]) & (labels['trip_id'] == row[1])]
        label = np.array(label, dtype=pd.Series)
        if math.isnan(label[0][2]) or label[0][2] > 4:
            continue
        # copy the speed, distance and time interval btw each pair of points X_orig

        # X_orig[i, 0:trip.shape[0]] = trip[['time_interval', 'distance_prev_point', 'speed']]
        X_orig[i, :, :, 0] = np.reshape(Channel_1, (n_H, n_W))
        X_orig[i, :, :, 1] = np.reshape(Channel_2, (n_H, n_W))
        X_orig[i, :, :, 2] = np.reshape(Channel_3, (n_H, n_W))


        # copy the the mode of transport to the Y_orig
        Y_orig[i] = int(label[0][2])

        i += 1

    #print('shape X-origin is:      ', X_orig.shape)
    #print('shape Y-origin is:      ', Y_orig.shape)
    #print('num_segements is:' , num_segements)
    # unique, counts = np.unique(Y_orig, return_counts=True)
    # print(dict(zip(unique, counts)))
    # Y_orig_to_csv = pd.DataFrame(Y_orig)
    # Y_orig_to_csv.to_csv("labels_data.csv")
    # X_orig_to_csv = pd.DataFrame(X_orig)
    # X_orig_to_csv.to_csv("segmented_data.csv")
    # # np.savetxt("labels_data.csv", Y_orig, delimiter=",")
    # # np.savetxt("segmented_data.csv", X_orig, delimiter=",")
    # print('the csvs are save')
    np.save("labels_data", Y_orig)
    np.save("segmented_data", X_orig)
    #pd.DataFrame(data=data[1:, 1:],  index = data[1:, 0], columns = data[0, 1:])


    return (X_orig, Y_orig)

    # X_train_flatten = points_segmented.reshape(X_orig.shape[0], -1).T
    # return (X_train_flatten)

######################flattening the channels######################
# def flattening_data(X_orig):
#     # Flatten the training and test sets
#     X_train_flatten = X_orig.reshape(X_orig.shape[0], -1).T
#     return (X_train_flatten)


######################split data to train-test######################
def split_train_test(X_origin, Y_orig):
    X_train_orig, X_test_orig, Y_train_orig, Y_test_orig = train_test_split(X_origin, Y_orig, test_size = 0.20, random_state = None)
    # print('shapes Y origin', Y_orig.shape)
    # indices = np.random.permutation(X_origin.shape[0])
    # print(' indices[80:] shape is:',  indices[80:].shape)
    # training_idx, test_idx = indices[80:], indices[:80]
    # X_train_orig, X_test_orig = X_origin[training_idx, :,:,:], X_origin[test_idx, :, :, :]
    # Y_train_orig, Y_train_orig = Y_orig[training_idx], Y_orig[test_idx]

    # training_idx = np.random.rand(X_origin.shape[0]) < 0.7
    # X_train_orig = X_origin[training_idx,:,:,: ]
    # X_test_orig = X_origin[~training_idx,:,:,:]
    # Y_train_orig = Y_orig[training_idx]
    # Y_test_orig = Y_orig[~training_idx]
    # print("shape is : ",X_train_orig.shape, X_test_orig.shape, Y_train_orig.shape, Y_test_orig.shape)
    # print("shape is : ", Y_test_orig)
    return (X_train_orig, X_test_orig, Y_train_orig, Y_test_orig)


######################Convert labes vector to one-hot######################
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


#############Create place holders for X and Y in tensorflow#################
def create_placeholders(n_H, n_W, num_channels, num_classes):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    ### START CODE HERE ### (approx. 2 lines)
    X = tf.placeholder(tf.float32, shape=(None, n_H, n_W, num_channels))
    Y = tf.placeholder(tf.float32, shape=[None, num_classes])

    return X, Y

################initialize the parameters#################
def initialize_parameters(weights):
    """
    Initializes weight parameters to build a neural network with tensorflow. For example, the shapes could be:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
                        W3 : ....
    Returns:
    parameters -- a dictionary of tensors containing W1, W2, W3 , ...
        """

    # define the parameters for conv layers
    parameters = {}

    for index, current_layer in enumerate(weights):
        # declare 'W's
        globals()['W{}'.format(index + 1)] = tf.get_variable('W{}'.format(index + 1),
                                                         current_layer,
                                                         initializer=tf.contrib.layers.xavier_initializer(seed=0))

        parameters['W{}'.format(index + 1)] = globals()['W{}'.format(index + 1)]

    return parameters

####################Forward propagation in tensorflow#########################
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> ........ -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    for index, param in enumerate(parameters):
        #print(param)
        #print(index, 'index is')
        #print('num_stride_conv2d:',num_stride_conv2d[index])
        #print('num_stride_maxpool:', num_stride_maxpool[index])
        # Retrieve the parameters from the dictionary "parameters"
        if index == 0:
            globals()['W{}'.format(index + 1)] = parameters['W{}'.format(index + 1)]

            # CONV2D: stride from num_stride_conv2d, padding 'SAME'
            globals()['Z{}'.format(index + 1)] = tf.nn.conv2d(X, globals()['W{}'.format(index + 1)]
                                                              , strides=[1, num_stride_conv2d[index], num_stride_conv2d[index], 1],
                                                              padding='SAME')

            # RELU
            globals()['A{}'.format(index + 1)]  = tf.nn.relu(globals()['Z{}'.format(index + 1)])

            # MAXPOOL: window size form stride from num_stride_maxpool, sride is the same size as window size, padding 'SAME'
            globals()['P{}'.format(index + 1)] = tf.nn.max_pool(globals()['A{}'.format(index + 1)], ksize=[1, num_stride_maxpool[index], num_stride_maxpool[index], 1],
                                strides=[1, num_stride_maxpool[index], num_stride_maxpool[index], 1],
                                padding='SAME')
        else:
            globals()['W{}'.format(index + 1)] = parameters['W{}'.format(index + 1)]

            # CONV2D: stride from num_stride_conv2d, padding 'SAME'
            globals()['Z{}'.format(index + 1)] = tf.nn.conv2d(globals()['P{}'.format(index)], globals()['W{}'.format(index + 1)]
                                                              , strides=[1, num_stride_conv2d[index],
                                                                         num_stride_conv2d[index], 1],
                                                              padding='SAME')

            # RELU
            globals()['A{}'.format(index + 1)] = tf.nn.relu(globals()['Z{}'.format(index + 1)])

            # MAXPOOL: window size form stride from num_stride_maxpool, sride is the same size as window size, padding 'SAME'
            globals()['P{}'.format(index + 1)] = tf.nn.max_pool(globals()['A{}'.format(index + 1)],
                                                                ksize=[1, num_stride_maxpool[index],
                                                                       num_stride_maxpool[index], 1],
                                                                strides=[1, num_stride_maxpool[index],
                                                                         num_stride_maxpool[index], 1],
                                                                padding='SAME')

    # FLATTEN
    globals()['P{}'.format(len(parameters))] = tf.contrib.layers.flatten(globals()['P{}'.format(len(parameters))])

    # FULLY-CONNECTED without non-linear activation function (softmax will be called later).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    globals()['Z{}'.format(len(parameters) + 1)] = tf.contrib.layers.fully_connected(globals()['P{}'.format(len(parameters))], num_classes, activation_fn=None)

    final_Z = globals()['Z{}'.format(len(parameters) + 1)]

    return final_Z


####################Computing Cost with softmax_cross_entropy in tensorflow#########################
def compute_cost(final_Z, Y):
    """
    Computes the cost

    Arguments:
    Final_Z -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as final_Z

    Returns:
    cost - Tensor of the cost function
    """

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_Z, labels=Y))

    return cost

####################Creates a list of random minibatches from (X, Y)#########################
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[1]  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    # print(m)
    # Y = np.transpose(Y)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    # print("too many",Y.shape[0], m)
    # print("Y.shape[0]",Y.shape[0],m)
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

####################Training the neural net model in Tensorflow#########################
def model(X_train, Y_train, X_test, Y_test, seg_size, learning_rate=0.009,
          num_epochs=1500, minibatch_size=2, print_cost=True):
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
    (m, n_H0, n_W0, n_C0) = X_train.shape

    n_y = Y_train.shape[1]
    costs = []

    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)


    # Initialize parameters
    parameters = initialize_parameters(weights)
    # Forward propagation: Build the forward propagation in the tensorflow graph
    final_Z = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(final_Z, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)
        _, total_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
        print("total cost is:  ", total_cost)
        # Do the training loop

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        predict_op = tf.argmax(final_Z, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        predictions, labels_train= sess.run([predict_op, tf.argmax(Y, 1)], feed_dict = {X: X_test, Y: Y_test})
        print(predictions, labels_train)
        # class_accuracy = tf.metrics.mean_per_class_accuracy(tf.argmax(Y, 1), predict_op, 5)
        # confusion_mat = class_accuracy.eval({Y: Y_test,X: X_test})


        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        confusion = tf.confusion_matrix(labels=tf.argmax(Y, 1), predictions=predict_op, num_classes=5)
        confusion_mat = confusion.eval({Y: Y_test,X: X_test})

        print(confusion_mat)

        #print(confusion_mat)

        # # Calculate the correct predictions
        # correct_prediction = tf.equal(tf.argmax(final_Z), tf.argmax(Y))
        #
        # # Calculate accuracy on the test set
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #
        # print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        # print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters

def main():
    # methoed 1: all points along trips feed to NN at once
    # padding_size= pad_size(points_table)
    # X_orig,Y_orig  = padding(points,padding_size,labels)
    # Method 2: segmented trips feed to NN
    points, labels, num_points_per_trip = read_db(points_table, labels_table)
    points_segmented = segmentation(points, seg_size, num_points_per_trip)
    X_orig, Y_orig = XY_preparation(points_segmented, labels, seg_size, num_channels)

    #load data from npy files
    #Y_orig = np.load('labels_data.npy')
    print(Y_orig.shape)
    #X_orig = np.load('segmented_data.npy')
    #Y_orig_df = pd.DataFrame(Y_orig, columns = ['mode'])


    #modes_num= Y_orig_df.groupby(['mode'])
    print(modes_num)
    time.sleep(100)
    #print(Y_orig.head)

    #for x in np.nditer(Y_orig):
    X_train, X_test, Y_train, Y_test = split_train_test(X_orig, Y_orig)
    Y_train = convert_to_one_hot(Y_train, 5)
    Y_test = convert_to_one_hot(Y_test, 5)
    parameters = model(X_train, Y_train, X_test, Y_test, seg_size)



    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
