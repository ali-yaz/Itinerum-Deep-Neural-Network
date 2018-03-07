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
from sklearn.preprocessing import normalize
import math
from sklearn.model_selection import train_test_split

#import matplotlib.pyplot as plt
#from sklearn.model_selection import KFold
#==============================================================================
__author__ = "Ali Yazdizadeh"
__date__ = "January 2018"
__email__ = "ali.yazdizadeh@mail.concordia.ca"
__python_version__ = "3.5.4"
#==============================================================================

start_time = time.time()

####################Setting  parameters#########################
#number of point per each segment
seg_size = 70
#number of nodes per each layer ('np


# .repeat' function is used)
#the default is 65 layers (without the output layer)
layer_nods = [70,22]
    #np.repeat([k for k in range(seg_size,5,-10)],1).tolist()
#number of classes, i.e. number of modes (here: 'walk','bike','car','public transit','car and public transit')
num_classes = 5
num_channels = 3

points_table = "segment_trip_id_cleaned_valid_btw_home_work_study"
labels_table = "mode_activity_trip_cleaned_valid_btw_home_work_study"

#############Make connection to the PostgreSQL database##########
db_conn = psycopg2.connect(user='postgres', password='postgresql', host='localhost', port=5432, database='MtlTrajet_tout_July2017')
db_conn.autocommit = True
db_cur = db_conn.cursor()



def read_db(points_table, labels_table):
    """

    :param point_table:
    :param labels_table:
    :return: pandas dataframes 'points', 'labels'
    """
    query = """SELECT * FROM {} where uuid = '8EEF93ED-5641-463D-947D-A8CBC2F0C57D' and trip_id = 29
    	

    """
    """
    order by uuid, trip_id, timestamp limit 24000
    
    
    select * from {} where mode in (
    select mode from {}
    group by mode having count(*) > 1
)
    
    where (uuid = '001DCAB0-2E98-42E2-85EB-CF297A3534EF'
    	and trip_id = 10) or (uuid = '001DCAB0-2E98-42E2-85EB-CF297A3534EF'
    	and trip_id = 19)
    	
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
    print('total number of points is:',num_points_per_trip['counts'].sum())

    return points, labels, num_points_per_trip

# ####################Read the point and labels data#########################
# points_table = "points"
# query = """SELECT * FROM {}
# where uuid in
# (select distinct(uuid) from {} order by uuid )
# order by uuid, trip_id, timestamp
# ;"""
# #'points','com_names' in follwoing commands are panda dataframe type
# points = pd.read_sql_query(query.format(points_table,points_table),con=db_conn)
#
# #read the column names of points table
# query ="""
# select column_name from information_schema.columns where table_name = '{}';
# """
# col_names = pd.read_sql_query(query.format(points_table),con=db_conn)
#
#
# #read the label data
# labels_table = "labels"
# query = "SELECT uid,trip_id,mode FROM {};"
# labels = pd.read_sql_query(query.format(labels_table),con=db_conn)
#
# #Calculate the number of points along each trip in pandas dataframe
# num_points_per_trip = points.groupby(['uuid', 'trip_id']).size().reset_index(name='counts')

#############create new dataframe with fixed size segments##########
def segmentation(points, seg_size,num_points_per_trip):
    i=0
    points_segmented = pd.DataFrame()
    #give same size for each segment
    for index, row in num_points_per_trip.iterrows():
        segment_counter = 0
        trip = points.loc[(points['uuid'] == row[0]) & (points['trip_id'] == row[1])]
        num_segs = math.ceil(row[2]/seg_size)

        padding = (-trip.shape[0]) % seg_size
        splitted_trip= np.array_split(np.concatenate((trip, np.zeros((padding, trip.shape[1])))), num_segs)
        for j in range(0, num_segs):
            segment_counter += 1
            trip = pd.DataFrame(data = splitted_trip[j],  columns=points.columns.values)
            trip = trip.assign(segment_id=segment_counter)
            points_segmented = points_segmented.append(trip, ignore_index=False)

        #for loop for splitting the points of a trip into seprate 'seg_size' segments.
        # for j in range(1,num_segs + 1):
        #
        #     segment_counter += 1
        #
        #     b_loc = (j-1)*seg_size
        #     e_loc = (j-1)*seg_size + (seg_size)
        #     if j == num_segs and row[2]%seg_size != 0:
        #         print('j is', j)
        #         print('segment_counter is', segment_counter)
        #         e_loc = row[2]
        #         trip = trip.assign(segment_id=segment_counter)
        #         temp = pd.DataFrame(0, index=np.arange(seg_size), columns=list(trip.columns.values))
        #         print(temp.shape)
        #         print(b_loc,e_loc)
        #         print(0,row[2]%seg_size)
        #         temp.iloc[0:row[2]%seg_size] = trip.iloc[b_loc:e_loc]
        #
        #         #print('temp is', temp)
        #         #print('trip is', trip)
        #         points_segmented = points_segmented.append(temp, ignore_index=False)
        #         print(points_segmented.shape)
        #         #print('points_segmented is:', points_segmented[b_loc:e_loc])
        #         print(temp)
        #         time.sleep(20)
        #
        #         continue
        #
        #     trip = trip.assign(segment_id=segment_counter)
        #     temp = pd.DataFrame(0, index=np.arange(seg_size), columns=list(trip.columns.values))
        #     temp.iloc[0:row[2] % seg_size] = trip.iloc[b_loc:e_loc]
        #     points_segmented = points_segmented.append(trip.iloc[b_loc:e_loc], ignore_index=False)
    #drop trips with na or zero values in 'uuid','trip_id','segment_id'
    points_segmented = points_segmented.dropna(subset=['uuid','trip_id','segment_id'])
    points_segmented = points_segmented[(points_segmented['uuid'] != 0) &
                                        (points_segmented['trip_id'] != 0) &
                                        (points_segmented['segment_id'] != 0)]



    return points_segmented

#############Preparing the X and Y data to feed to neural net##########
def XY_preparation(points_segmented, labels, seg_size,num_channels):
    # Flatten the training and test sets
    num_segements = points_segmented.drop_duplicates(subset=('uuid','trip_id', 'segment_id'), keep='first', inplace=False).shape[0]
    uuid_trip_id_segments = points_segmented.drop_duplicates(subset=('uuid','trip_id', 'segment_id'), keep='first', inplace=False)[['uuid','trip_id', 'segment_id']]

    #creating X_orig and Y_orig arrays
    X_orig = np.zeros((num_segements, seg_size, num_channels))
    X_orig[X_orig == 0] = -20000000
    Y_orig = np.zeros(num_segements, dtype=int)
    #number of points along each trip
    #print(uuid_trip_id_segments.shape)
    i = 0
    #assign the label for each trip
    for index, row in uuid_trip_id_segments.iterrows():
        #select all the points for each segment
        trip = points_segmented.loc[(points_segmented['uuid'] == row[0]) &
                                    (points_segmented['trip_id'] == row[1]) &
                                    (points_segmented['segment_id'] == row[2])]
        #aasing the labels to each segment
        label = labels.loc[(labels['uid'] == row[0]) & (labels['trip_id'] == row[1])]
        label = np.array(label, dtype=pd.Series)

        if math.isnan(label[0][2]) or label[0][2] > 4:
            continue
        #copy the speed, distance and time interval btw each pair of points X_orig
        X_orig[i,0:trip.shape[0]] = trip[['time_interval','distance_prev_point','speed']]

        # copy the the mode of transport to the Y_orig
        Y_orig[i] = int(trip['mode'][0])
        i += 1

    # np.save("DNN_labels_data", Y_orig)
    # np.save("DNN_segmented_data", X_orig)

    return(X_orig,Y_orig)


    # X_train_flatten = points_segmented.reshape(X_orig.shape[0], -1).T
    # return (X_train_flatten)

############################padding all the trips to the same size as the longets trip############################
###NOTE: follwoing 2 functions, i.e. "pad_size" and "padding", are not used while "segmentation" function is used.
def pad_size (points_table):
    """find the max num of points along a trip which is padding size
    points_table   the name of points table in PostgreSQL db
    """
    query ="""
    SELECT * FROM
    (SELECT count(*) as num FROM {} GROUP BY uuid, trip_id) a
    ORDER BY num DESC LIMIT 1;
    """
    db_cur.execute(query.format(points_table))
    padding_size = db_cur.fetchone()
    padding_size = padding_size[0]
    print("padding size is:",padding_size)

    query = """
        SELECT avg(num) FROM
        (SELECT count(*) as num FROM {} GROUP BY uuid, trip_id) a
        ;
        """
    db_cur.execute(query.format(points_table))
    average_points = db_cur.fetchone()
    print("average num of points is:", average_points[0])
    return (padding_size)

def padding(points, padding_size,labels):
    """
    :Goal: #make the padding array for GPS points and
    :param points:
    :return: the padded points
    :labels: the labels of the classes
    """
    #create an array of zeros


    num_obs = points.shape[0]
    num_col = points.shape[1]
    print("number of points:",num_obs)
    print("number of columns",num_col)

    #points_array = np.zeros(num_obs,num_col)

    #convert list to panda dataframe

    #findig the number of trips
    query ="""
    SELECT COUNT(DISTINCT(uuid, trip_id)) FROM {};
    """
    db_cur.execute(query.format(points_table))
    num_trips = db_cur.fetchone()
    num_trips = num_trips[0]
    print("number of trips =", num_trips)
    #creating X_orig and Y_orig arrays
    X_orig = np.zeros((num_trips, padding_size, 3))
    X_orig[X_orig == 0] = -20000000
    Y_orig = np.zeros(num_trips, dtype=int)
    #print(X_orig[1])

    #number of points along each trip
    query ="""
    select uuid, trip_id, count(*) from {} group by uuid, trip_id
    """
    num_points_per_trip = pd.read_sql_query(query.format(points_table), con=db_conn)

    # db_cur.execute(query.format(points_table))
    # num_points_per_trip = db_cur.fetchall()
    # num_points_per_trip = np.asanyarray(num_points_per_trip)

    i = 0
    #print(num_points_per_trip.iloc[0])
    #assign the label for each trip
    for index, row in num_points_per_trip.iterrows():
        i += 1
        trip = points.loc[(points['uuid'] == row[0]) & (points['trip_id'] == row[1])]
        label = labels.loc[(labels['uid'] == row[0]) & (labels['trip_id'] == row[1])]
        label = np.array(label, dtype=pd.Series)
        if math.isnan(label[0][2]) or label[0][2] > 4:
            continue
        #print("label is:", label[0][2])
        #print("label mode is", label['mode'])


        #copy the speed, distance and time interval btw each pair of points X_orig
        X_orig[index,0:trip.shape[0]] = trip[['time_interval','distance_prev_point','speed']]

        # copy the the mode of transport to the Y_orig

        Y_orig[index] = int(label[0][2])

    return(X_orig,Y_orig)

######################flattening the channels######################
def flattening_data(X_orig):
    # Flatten the training and test sets

    # X_orig[:, :, 1] = normalize(X_orig[:, :, 1], axis=0).ravel()
    # X_orig[:, :, 2] = normalize(X_orig[:, :, 2], axis=1).ravel()
    # X_orig[:, :, 3] = normalize(X_orig[:, :, 3], axis=1).ravel()
    # print(X_orig)
    # time.sleep(100)
    print('X_orig.shape[0]', X_orig.shape[0])
    #X_train_flatten = X_orig.flatten('F').T

    X_train_flatten = X_orig.reshape(X_orig.shape[0], -1).T
    #print('X_orig',type(X_orig))
    #print('X_train_flatten',X_train_flatten)

    return (X_train_flatten)

#####################split data to train-test######################
# def split_train_test(X_flatten, Y_orig):
#     training_idx = np.random.rand(X_flatten.shape[1]) < 0.8
#     X_train_orig = X_flatten[:,training_idx]
#     X_test_orig = X_flatten[:, ~training_idx]
#     Y_train_orig = Y_orig[training_idx]
#     Y_test_orig = Y_orig[~training_idx]
#     #print("shape is : ",X_train_orig.shape, X_test_orig.shape, Y_train_orig.shape, Y_test_orig.shape)
#     #print("shape is : ", Y_test_orig)
#     return (X_train_orig, X_test_orig, Y_train_orig, Y_test_orig)

def split_train_test(X_train_flatten, Y_orig):
    #print(X_train_flatten.shape)
    #print(Y_orig.shape)
    X_train_orig, X_test_orig, Y_train_orig, Y_test_orig = train_test_split(X_train_flatten.T, Y_orig, test_size = 0.20, random_state = None)
    print('shape X_train_orig.T',X_train_orig,X_train_orig.T.shape)
    print('shape X_test_orig.T',X_test_orig,X_test_orig.T.shape)

    return (X_train_orig.T, X_test_orig.T, Y_train_orig, Y_test_orig)

######################Convert labes vector to one-hot######################
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

#############Create place holders for X and Y in tensorflow#################
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
      

    return X, Y

################initialize the parameters#################
def initialize_parameters(layer_nods,num_classes,seg_size):
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
    #tf.set_random_seed(1)  # so that your "random" numbers match ours

    # define the number of nodes for each layer, except the last one
    # add the number of layers for the first and last column
    layer_nods.insert(0, 3 * seg_size)
    #layer_nods = np.insert(layer_nods,0,3 * seg_size)
    layer_nods.append(num_classes)
    #layer_nods = np.append(layer_nods,num_classes)
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

####################Forward propagation in tensorflow#########################
def forward_propagation(X, parameters):
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
                #e.g.: Z1 = np.dot(W1, X) + b1
                globals()['Z{}'.format(index)] = \
                    tf.add(tf.matmul(globals()['W{}'.format(index)], X),
                           globals()['b{}'.format(index)])
                # e.g.:# A1 = relu(Z1)
                globals()['A{}'.format(index)] = tf.nn.relu(globals()['Z{}'.format(index)])

            else:
                # e.g.: Z2 = np.dot(W2, a1) + b2
                globals()['Z{}'.format(index)] = \
                    tf.add(tf.matmul(globals()['W{}'.format(index)],
                                     globals()['A{}'.format(index - 1)]),
                           globals()['b{}'.format(index)])
                # e.g.:# A2 = relu(Z2)
                globals()['A{}'.format(index)] = tf.nn.relu(globals()['Z{}'.format(index)])

    final_Z = globals()['Z{}'.format(len(layer_nods) - 1)]
    return final_Z

####################Computing Cost with softmax_cross_entropy in tensorflow#########################
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

    #calculating the cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

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
    #print(m)
    #Y = np.transpose(Y)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    #print("too many",Y.shape[0], m)
    #print("Y.shape[0]",Y.shape[0],m)
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

####################This function is used for predicting individual sample, not used now#########################
def predict(X, parameters,seg_size):
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    W4 = tf.convert_to_tensor(parameters["W4"])
    b4 = tf.convert_to_tensor(parameters["b4"])
    W5 = tf.convert_to_tensor(parameters["W5"])
    b5 = tf.convert_to_tensor(parameters["b5"])
    W6 = tf.convert_to_tensor(parameters["W6"])
    b6 = tf.convert_to_tensor(parameters["b6"])

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3,
              "W3": W4,
              "b3": b4,
              "W3": W5,
              "b3": b5,
              "W3": W6,
              "b3": b6}

    x = tf.placeholder("float", [3*seg_size, 1])

    z6 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z6)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict={x: X})

    return prediction

####################Training the neural net model in Tensorflow#########################
def model(X_train, Y_train, X_test, Y_test, seg_size, learning_rate=0.0001,
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
    (n_x, m) = X_train.shape # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]  # n_y : output size
    costs = []  # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)
      

    # Initialize parameters
    parameters = initialize_parameters(layer_nods,num_classes,seg_size)


    # Forward propagation: Build the forward propagation in the tensorflow graph
    final_Z = forward_propagation(X, parameters)
    print(final_Z)
     

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
        _, total_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
        #print(parameters)
        print("total cost is:  " ,total_cost)
        # Do the training loop

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        correct_prediction = tf.equal(tf.argmax(final_Z), tf.argmax(Y))

        predict_op = tf.argmax(final_Z)

        predictions, labels_train= sess.run([predict_op, tf.argmax(Y)], feed_dict = {X: X_test, Y: Y_test})
        print(predictions, labels_train)

        # Calculate the correct predictions
        #correct_prediction = tf.equal(tf.argmax(final_Z), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        confusion = tf.confusion_matrix(labels=tf.argmax(Y), predictions=predict_op, num_classes=5)
        confusion_mat = confusion.eval({Y: Y_test,X: X_test})

        print(confusion_mat)

        return parameters

###Currently not used
#model with batch minimization
# def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
#           num_epochs=1500, minibatch_size=2, print_cost=True):
#     """
#     Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
#
#     Arguments:
#     X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
#     Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
#     X_test -- training set, of shape (input size = 12288, number of training examples = 120)
#     Y_test -- test set, of shape (output size = 6, number of test examples = 120)
#     learning_rate -- learning rate of the optimization
#     num_epochs -- number of epochs of the optimization loop
#     minibatch_size -- size of a minibatch
#     print_cost -- True to print the cost every 100 epochs
#
#     Returns:
#     parameters -- parameters learnt by the model. They can then be used to predict.
#     """
#
#     ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
#     tf.set_random_seed(1)  # to keep consistent results
#     seed = 3  # to keep consistent results
#     (n_x, m) = X_train.shape # (n_x: input size, m : number of examples in the train set)
#     n_y = Y_train.shape[0]  # n_y : output size
#     #print("X_train.shape",X_train.shape,Y_train.shape)
#     costs = []  # To keep track of the cost
#
#     # Create Placeholders of shape (n_x, n_y)
#       
#     X, Y = create_placeholders(n_x, n_y)
#       
#
#     # Initialize parameters
#       
#     parameters = initialize_parameters(padding_size)
#       
#
#     # Forward propagation: Build the forward propagation in the tensorflow graph
#       
#     Z3 = forward_propagation(X, parameters)
#       
#
#     # Cost function: Add cost function to tensorflow graph
#       
#     cost = compute_cost(Z3, Y)
#       
#
#     # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
#       
#     optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#       
#
#     # Initialize all the variables
#     init = tf.global_variables_initializer()
#
#     # Start the session to compute the tensorflow graph
#     with tf.Session() as sess:
#
#         # Run the initialization
#         sess.run(init)
#
#         # Do the training loop
#         for epoch in range(num_epochs):
#
#             epoch_cost = 0.  # Defines a cost related to an epoch
#             num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
#             seed = seed + 1
#             minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
#
#             for minibatch in minibatches:
#                 # Select a minibatch
#                 (minibatch_X, minibatch_Y) = minibatch
#
#                 # IMPORTANT: The line that runs the graph on a minibatch.
#                 # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
#                   
#                 _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
#                   
#
#                 epoch_cost += minibatch_cost / num_minibatches
#
#             # Print the cost every epoch
#             if print_cost == True and epoch % 100 == 0:
#                 print("Cost after epoch %i: %f" % (epoch, epoch_cost))
#             if print_cost == True and epoch % 5 == 0:
#                 costs.append(epoch_cost)
#
#         # plot the cost
#         plt.plot(np.squeeze(costs))
#         plt.ylabel('cost')
#         plt.xlabel('iterations (per tens)')
#         plt.title("Learning rate =" + str(learning_rate))
#         plt.show()
#
#         # lets save the parameters in a variable
#         parameters = sess.run(parameters)
#         print("Parameters have been trained!")
#
#         # Calculate the correct predictions
#         correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
#
#         # Calculate accuracy on the test set
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#
#         print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
#         print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
#
#         return parameters


def main():
    # methoed 1: all points along trips feed to NN at once
    # padding_size= pad_size(points_table)
    # X_orig,Y_orig  = padding(points,padding_size,labels)

    # Method 2: segmented trips feed to NN
    points, labels, num_points_per_trip = read_db(points_table, labels_table)
    points_segmented = segmentation(points, seg_size, num_points_per_trip)
    X_orig, Y_orig = XY_preparation(points_segmented, labels, seg_size, num_channels)

    print(X_orig.shape, Y_orig.shape)

    #load data from npy files
    # Y_orig = np.load('DNN_labels_data.npy')
    # X_orig = np.load('DNN_segmented_data.npy')
    #print(X_orig.shape, Y_orig.shape)
    X_flatten = flattening_data(X_orig)
    X_train, X_test, Y_train, Y_test = split_train_test(X_flatten, Y_orig)
    #print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    Y_train = convert_to_one_hot(Y_train, 5)
    Y_test = convert_to_one_hot(Y_test, 5)
    parameters = model(X_test, Y_test, X_test, Y_test, seg_size)

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()