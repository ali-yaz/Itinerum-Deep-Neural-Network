import numpy as np
import time
import tensorflow as tf
import math
import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from Kernel_smoothing import savitzky_golay

#############Make connection to the PostgreSQL database##########
db_conn = psycopg2.connect(user='postgres', password='postgresql', host='localhost', port=5432,
                           database='MtlTrajet_tout_July2017')
db_conn.autocommit = True
db_cur = db_conn.cursor()
#############Make connection to the PostgreSQL database##########



# points_table = "segment_trip_id_cleaned_valid_btw_home_work_study"
points_table = "segment_trip_id_cleaned_valid_btw_home_work_study_joined_morethanfive"
labels_table = "mode_activity_trip_cleaned_valid_btw_home_work_study"


def read_db(points_table, labels_table):
    """

    :param point_table:
    :param labels_table:
    :return: pandas dataframes 'points', 'labels'
    """
    query = """SELECT * {}
    where rank > 3
      AND cumulative_distance > 250
      AND distance_prev_point > 0
      AND ((mode = 0 and speed <= 7 and acceleration <= 3)
       OR (mode = 1 and speed <= 12 and acceleration <= 3)
        OR (mode = 2 and speed <= 34 and acceleration <= 4)
         OR (mode = 3 and speed <= 50 and acceleration <= 10))
    ORDER BY uuid, trip_id, timestamp

    
    """
    """
    SELECT * {}
    where rank > 3
      AND cumulative_distance > 250
      AND distance_prev_point > 0
      AND ((mode = 0 and speed <= 7 and acceleration <= 3)
       OR (mode = 1 and speed <= 12 and acceleration <= 3)
        OR (mode = 2 and speed <= 34 and acceleration <= 4)
         OR (mode = 3 and speed <= 50 and acceleration <= 10))
    ORDER BY uuid, trip_id, timestamp
    
    WHERE (uuid,trip_id) IN (SELECT uuid, trip_id FROM mode_activity_trip_cleaned_valid_btw_home_work_study_ranked_more_than_five_trips 
    where rank > 5 AND cumulative_distance > 250) 
    AND ((mode = 0 and speed <= 7 and acceleration <= 3)
     OR (mode = 1 and speed <= 12 and acceleration <= 3)
      OR (mode = 2 and speed <= 34 and acceleration <= 4)
       OR (mode = 3 and speed <= 50) and acceleration <= 10 )
    ORDER BY uuid, trip_id, timestamp
    
    
    order by uuid, trip_id, timestamp limit 10000
    
    where (uuid = '001DCAB0-2E98-42E2-85EB-CF297A3534EF'
    	and trip_id = 10) ORDER BY uuid, trip_id, timestamp
   
    
    where (uuid,trip_id) in (SELECT uuid, trip_id FROM mode_activity_trip_cleaned_valid_btw_home_work_study_ranked_more_than_five_trips 
    where rank > 5)
    order by uuid, trip_id, timestamp limit 24000

    where (uuid = '001DCAB0-2E98-42E2-85EB-CF297A3534EF'
    	and trip_id = 10)
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
    # 'points','com_names' in follwoing commands are panda dataframe type
    points = pd.read_sql_query(query.format(points_table), con=db_conn)

    # read the label data
    query = "SELECT uid,trip_id,mode FROM {};"
    labels = pd.read_sql_query(query.format(labels_table), con=db_conn)

    num_points_per_trip = points.groupby(['uuid', 'trip_id']).size().reset_index(name='counts')
    print('total number of points is:', num_points_per_trip['counts'].sum())

    return points, labels, num_points_per_trip

def filtering(points, num_points_per_trip, windows_size = 15):
    """
    :param X_orig:
    :return: X_orig filtered high errors with savitzky_golay alg.
    """
    for index, row in num_points_per_trip.iterrows():
        if row[2] < windows_size:
            continue
        temp_speed = np.copy(points.loc[(points['uuid'] == row[0]) & (points['trip_id'] == row[1])]['speed'].values)
        temp_acceleration = np.copy(points.loc[(points['uuid'] == row[0]) & (points['trip_id'] == row[1])]['acceleration'].values)
        temp_jerk = np.copy(points.loc[(points['uuid'] == row[0]) & (points['trip_id'] == row[1])]['jerk'].values)
        temp_bearing_rate = np.copy(points.loc[(points['uuid'] == row[0]) & (points['trip_id'] == row[1])]['bearing_rate'].values)

        points.loc[(points['uuid'] == row[0]) & (points['trip_id'] == row[1]), 'speed'] = \
            savitzky_golay(temp_speed, window_size = 15, order=4)
        points.loc[(points['uuid'] == row[0]) & (points['trip_id'] == row[1]), 'acceleration'] = \
            savitzky_golay(temp_acceleration, window_size = 15, order=4)
        points.loc[(points['uuid'] == row[0]) & (points['trip_id'] == row[1]), 'jerk'] = \
            savitzky_golay(temp_jerk, window_size = 15, order=4)
        points.loc[(points['uuid'] == row[0]) & (points['trip_id'] == row[1]), 'bearing_rate'] = \
            savitzky_golay(temp_bearing_rate, window_size = 15, order=4)
    # for filtering segments instead of whole trips
    # for obs in range(0,X_orig.shape[0]):
    #     for var in range(0,num_channels):
    #         temp = np.copy(X_orig[obs, :, var])
    #         print(temp)
    #         X_orig[obs, :, var] = savitzky_golay(X_orig[obs, :, var], window_size= 15, order=4)
    #         print(X_orig[obs, :, var])
    #         print(X_orig[obs, :, var] - temp)
    #         time.sleep(100000)
    return points


# if var == 3:
#     print(X_orig[obs,:,var])
#     t = np.linspace(-4, 4, 70)
#     plt.plot(t, temp, 'k', lw=1.5, label='Original signal')
#     plt.plot(t, temp2, 'r', label='Filtered signal')
#     plt.legend()
#     plt.show()
#     time.sleep(1000)

def normalizing(X_orig):
    """

    :param X_orig:
    :return: X_orig
    """
    # print(X_orig.shape)
    # X_orig = np.around(X_orig, 8)
    # X_orig = np.nan_to_num(X_orig, copy=False)
    # # print(X_orig[5, :, :])
    for i in range(0,X_orig.shape[2]):
        X_orig[:, :, i] = preprocessing.normalize(X_orig[:, :, i])

    # for i in range(0,X_orig.shape[0]):
    #      X_orig[i,:,:] = preprocessing.scale(X_orig[i,:,:])
    #print(X_orig[5, :, :])

    return X_orig

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
def segmentation(points, seg_size, num_points_per_trip):
    i = 0
    points_segmented = pd.DataFrame()
    # give same size for each segment
    for index, row in num_points_per_trip.iterrows():
        segment_counter = 0
        trip = points.loc[(points['uuid'] == row[0]) & (points['trip_id'] == row[1])]
        num_segs = math.ceil(row[2] / seg_size)

        padding = (-trip.shape[0]) % seg_size
        splitted_trip = np.array_split(np.concatenate((trip, np.zeros((padding, trip.shape[1])))), num_segs)
        for j in range(0, num_segs):
            segment_counter += 1
            trip = pd.DataFrame(data=splitted_trip[j], columns=points.columns.values)
            trip = trip.assign(segment_id=segment_counter)
            points_segmented = points_segmented.append(trip, ignore_index=False)

            # for loop for splitting the points of a trip into seprate 'seg_size' segments.
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
    # drop trips with na or zero values in 'uuid','trip_id','segment_id'
    points_segmented = points_segmented.dropna(subset=['uuid', 'trip_id', 'segment_id'])
    points_segmented = points_segmented[(points_segmented['uuid'] != 0) &
                                        (points_segmented['trip_id'] != 0) &
                                        (points_segmented['segment_id'] != 0)]

    return points_segmented


#############Preparing the X and Y data to feed to neural net##########
def XY_preparation(points_segmented, labels, seg_size, num_channels):
    #
    num_segements = \
    points_segmented.drop_duplicates(subset=('uuid', 'trip_id', 'segment_id'), keep='first', inplace=False).shape[0]
    print('Number of segments is:', num_segements)
    uuid_trip_id_segments = \
    points_segmented.drop_duplicates(subset=('uuid', 'trip_id', 'segment_id'), keep='first', inplace=False)[
        ['uuid', 'trip_id', 'segment_id']]

    # creating X_orig and Y_orig arrays
    X_orig = np.zeros((num_segements, seg_size, num_channels))
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

        # aasing the labels to each segment
        label = labels.loc[(labels['uid'] == row[0]) & (labels['trip_id'] == row[1])]
        label = np.array(label, dtype=pd.Series)

        if math.isnan(label[0][2]) or label[0][2] > 3:
            continue
        # copy the speed, distance and time interval btw each pair of points X_orig
        X_orig[i, 0:trip.shape[0]] = \
            trip[['speed', 'acceleration', 'jerk', 'bearing_rate', 'avg_dist_btw_points',
                  'cumulative_dist_btw_points', ]]
                  'cumulative_dist_btw_points', ]]
        # 'time_interval', 'distance_prev_point',
        # , 'acceleration', 'jerk'
        #,'acceleration'

        # copy the the mode of transport to the Y_orig
        Y_orig[i] = int(trip['mode'][0])
        i += 1
    X_orig = np.nan_to_num(X_orig, copy=False)
    #print(np.isinf(X_orig))
    #
    np.save("DNN_labels_data", Y_orig)
    np.save("DNN_segmented_data", X_orig)
    #
    print('files are saved')
    #time.sleep(1000)


    return (X_orig, Y_orig)

def split_train_test(X_train_flatten, Y_orig):
    #print(X_train_flatten.shape)
    #print(Y_orig.shape)
    X_train_orig, X_test_orig, Y_train_orig, Y_test_orig = train_test_split(X_train_flatten.T, Y_orig, test_size = 0.20, random_state = None)
    #print('shape X_train_orig.T',X_train_orig,X_train_orig.T.shape)
    #print('shape X_test_orig.T',X_test_orig,X_test_orig.T.shape)

    return (X_train_orig.T, X_test_orig.T, Y_train_orig, Y_test_orig)

######################Convert labes vector to one-hot######################
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

######################flattening the channels######################
def flattening_data(X_orig):
    # Flatten the training and test sets

    # X_orig[:, :, 1] = normalize(X_orig[:, :, 1], axis=0).ravel()
    # X_orig[:, :, 2] = normalize(X_orig[:, :, 2], axis=1).ravel()
    # X_orig[:, :, 3] = normalize(X_orig[:, :, 3], axis=1).ravel()
    # print(X_orig)
    # print('X_orig.shape[0]', X_orig.shape)
    # time.sleep(10000)
    #X_train_flatten = X_orig.flatten('F').T

    # print('X_orig.shape[0]', X_orig.shape[0])

    X_train_flatten = X_orig.reshape((X_orig.shape[0], -1), order='F').T
    #print('X_train_flatten',X_train_flatten.shape)
    #print(X_train_flatten[:,5])
    return (X_train_flatten)

# def main():
#     points, labels, num_points_per_trip = read_db(points_table, labels_table)
#     points = filtering(points, num_points_per_trip)
#     points_segmented = segmentation(points, seg_size, num_points_per_trip)
#     X_orig , Y_orig = XY_preparation(points_segmented, labels, seg_size, num_channels)
#     normalizing(X_orig)
#     #X_orig = normilizing(X_orig)
#     X_flatten = flattening_data(X_orig)
#     X_train, X_test, Y_train, Y_test = split_train_test(X_flatten, Y_orig)
#     Y_train = convert_to_one_hot(Y_train, 5)
#     Y_test = convert_to_one_hot(Y_test, 5)
#
# if __name__ == "__main__":
#     ###################Setting  parameters#########################
#     # number of point per each segment
#     seg_size = 70
#     # number of nodes per each layer ('np
#
#
#     # .repeat' function is used)
#     # the default is 65 layers (without the output layer)
#     layer_nods = np.repeat([k for k in range(3 * seg_size, 0, -5)], 2).tolist()
#     # [seg_size, 40, 10]
#     # np.repeat([k for k in range(seg_size,5,-10)],1).tolist()
#     # number of classes, i.e. number of modes (here: 'walk','bike','car','public transit','car and public transit')
#     num_classes = 4
#     num_channels = 4
#     main()
