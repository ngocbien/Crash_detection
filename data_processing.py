#!/usr/local/bin/python3
import config
import dateutil.parser as parser
import numpy as np
import utils
from pykalman import KalmanFilter
def get_raw_data(raw_data):
    """
    :param raw_data: is a dictionnary {'datetime': datetime, 'accelerometer': [{'x': acc_x, 'y': acc_y,
    'z': acc_z}], 'gyroscope': [{'x': gyro_x, 'y': gyro_y, 'z': gyro_z}]}
    :return: numpy array 7 axe acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, date_time
    """
    index_count = 0
    for line in raw_data:
        date_time = line['datetime']
        date_time = parser.parse(date_time)
        acc = line['accelerometer'][0]
        acc_x = float(acc['x'])/config.SENSITIVITY_ACC
        acc_y = float(acc['y'])/config.SENSITIVITY_ACC
        acc_z = float(acc['z'])/config.SENSITIVITY_ACC
        gyro = line['gyroscope'][0]
        gyro_x = float(gyro['x'])/config.SENSITIVITY_GYRO
        gyro_y = float(gyro['y'])/config.SENSITIVITY_GYRO
        gyro_z = float(gyro['z'])/config.SENSITIVITY_GYRO
        gyro_x = utils.degree_to_radian(gyro_x)
        gyro_y = utils.degree_to_radian(gyro_y)
        gyro_z = utils.degree_to_radian(gyro_z)
        new_row = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, date_time]
        if index_count == 0:
            data_processed = np.asanyarray(new_row)
        else:
            data_processed = np.vstack([data_processed, new_row])
        index_count += 1
    return data_processed

def reduce_bias(raw_data):
    """
    :param raw_data: is a dictionnary {'datetime': datetime, 'accelerometer': [{'x': acc_x, 'y': acc_y,
    'z': acc_z}], 'gyroscope': [{'x': gyro_x, 'y': gyro_y, 'z': gyro_z}]}
    :return: data_processed with 4 axes, acc_x, acc_y , acc_z , datetime
    """
    data_processed = get_raw_data(raw_data)
    data_processed = data_processed[:, [0, 1, 2, 6]]
    data_processed[:, :3] = data_processed[:, :3] - config.OFFSET
    return data_processed

def processing_data_by_agreated(raw_data, M = 1):
    """
    :param raw_data: numpy array of three axes
    :param M: integer, number of line to agreate example 1, 2, 3, ...
    :return: numpy array data processed
    """
    first_processed_data = reduce_bias(raw_data) # have 4 axes acc_x, acc_y, acc_z and datetime
    if M == 1:
        return first_processed_data
    N = len(first_processed_data[:,0])//M
    for index in range(N):
        from_index = index * M
        this_row = first_processed_data[from_index, :3]
        for i in range(M-1):
            this_row += first_processed_data[from_index + i + 1, :3]/M
        this_row = np.append(this_row, first_processed_data[index * M, 3])
        if index == 0:
            data_processed = np.asarray(this_row)
        else:
            data_processed = np.vstack([data_processed, this_row])
    return data_processed

def processing_by_kalman_filter(raw_data):
    """
    :param raw_data: numpy array 7 axes acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, date_time
    acc_x, acc_y , acc_z is in radian( and not in degree). Use utils.degre_to_radian() function to
    cover from degree to radian
    :return: numpy array 4 axes: acc_x, acc_y, acc_z , datetime
    """
    random_state = np.random.RandomState(0)
    delta_t = 0.001
    transition_matrix = [[1, delta_t, delta_t, delta_t, delta_t, delta_t],
                         [delta_t, 1, delta_t, delta_t, delta_t, delta_t],
                         [delta_t, delta_t, 1, delta_t, delta_t, delta_t],
                         [delta_t, delta_t, delta_t, 1, delta_t, delta_t],
                         [delta_t, delta_t, delta_t, delta_t, 1, delta_t],
                         [delta_t, delta_t, delta_t, delta_t, delta_t, 1]]
    transition_offset = [delta_t, delta_t, delta_t, delta_t, delta_t, delta_t]
    observation_matrix = np.eye(6) + random_state.randn(6, 6) * 0.01
    observation_offset = [0.019777696141400954, -0.02717658649906955, 0.8597378069103433, .0, .0, .0]
    transition_covariance = np.eye(6)
    observation_covariance = np.eye(6) + random_state.randn(6, 6) * 0.01
    # initial_state_mean = [5, -5]
    initial_state_mean = [.0, .0, .0, .0, .0, .0]
    initial_state_covariance = transition_matrix

    # sample from model
    kf = KalmanFilter(
        transition_matrix, observation_matrix, transition_covariance,
        observation_covariance, transition_offset, observation_offset,
        initial_state_mean, initial_state_covariance,
        random_state=random_state
    )
    observations = raw_data[:, :6]
    #print(raw_data[0, :])
    #estimate from raw_data using kalman smooth method
    acc_estimates = kf.smooth(observations)[0]
    raw_data[:, :3] = acc_estimates[:, :3]
    raw_data = raw_data[:, [0, 1, 2, 6]]
    #print(raw_data[0, :])
    return raw_data



