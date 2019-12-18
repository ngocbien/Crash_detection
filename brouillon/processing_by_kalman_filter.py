#!/usr/local/bin/python3
#import dummy_project.config as config
import dateutil.parser as parser
import numpy as np
from numpy import dot, sum, tile, linalg
from numpy.linalg import inv
import sys
sys.path.insert(0, '/Users/NhatMinh/AVICEN/dummy_project/')

import config
import pykalman

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

def kf_update(X, P, Y, H, R):

    IM = dot(H, X)
    IS = R + dot(H, dot(P, H.T))
    K = dot(P, dot(H.T, inv(IS)))
    X = X + dot(K, (Y-IM))
    P = P - dot(K, dot(IS, K.T))
    LH = gauss_pdf(Y, IM, IS)
    return (X,P,K,IM,IS,LH)

def gauss_pdf(X, M, S):

    if M.shape()[1] == 1:
        DX = X - tile(M, X.shape()[1])
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
        P = exp(-E)
    elif X.shape()[1] == 1:
        DX = tile(X, M.shape()[1])- M
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
        P = exp(-E)
    else:
        DX = X-M
        E = 0.5 * dot(DX.T, dot(inv(S), DX))
        E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
        P = exp(-E)
    return (P[0],E[0])

def test_kalman_filter():
    from pykalman import KalmanFilter
    import numpy as np
    kf = KalmanFilter(transition_matrices=[[1, 1], [0, 1]], observation_matrices=[[0.1, 0.5], [-0.3, 0.0]])
    measurements = np.asarray([[1, 0], [0, 0], [0, 1]])  # 3 observations
    kf = kf.em(measurements, n_iter=5)
    (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
    (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

def test_kalman2():
    from pykalman import UnscentedKalmanFilter
    ukf = UnscentedKalmanFilter(lambda x, w: x + np.sin(w), lambda x, v: x + v, transition_covariance=0.1)
    (filtered_state_means, filtered_state_covariances) = ukf.filter([0, 1, 2])
    (smoothed_state_means, smoothed_state_covariances) = ukf.smooth([0, 1, 2])



import numpy as np
import pylab as pl
from pykalman import UnscentedKalmanFilter

# initialize parameters
def transition_function(state, noise):
    a = np.sin(state[0]) + state[1] * noise[0]
    b = state[1] + noise[1]
    return np.array([a, b])

def observation_function(state, noise):
    C = np.array([[-1, 0.5], [0.2, 0.1]])
    return np.dot(C, state) + noise

transition_covariance = np.eye(2)
random_state = np.random.RandomState(0)
observation_covariance = np.eye(2) + random_state.randn(2, 2) * 0.1
initial_state_mean = [0, 0]
initial_state_covariance = [[1, 0.1], [-0.1, 1]]

# sample from model
kf = UnscentedKalmanFilter(
    transition_function, observation_function,
    transition_covariance, observation_covariance,
    initial_state_mean, initial_state_covariance,
    random_state=random_state
)
states, observations = kf.sample(50, initial_state_mean)
print(type(observations))
print(observations[0,0])
print(observations.shape)
# estimate state with filtering and smoothing
filtered_state_estimates = kf.filter(observations)[0]
smoothed_state_estimates = kf.smooth(observations)[0]

# draw estimates
pl.figure()
lines_true = pl.plot(states, color='b')
lines_filt = pl.plot(filtered_state_estimates, color='r', ls='-')
lines_smooth = pl.plot(smoothed_state_estimates, color='g', ls='-.')
pl.legend((lines_true[0], lines_filt[0], lines_smooth[0]),
          ('true', 'filt', 'smooth'),
          loc='lower left'
)
pl.show()


import numpy as np
import pylab as pl
from pykalman import KalmanFilter

# specify parameters
random_state = np.random.RandomState(0)
transition_matrix = [[1, 0.1], [0, 1]]
transition_offset = [-0.1, 0.1]
observation_matrix = np.eye(2) + random_state.randn(2, 2) * 0.1
observation_offset = [1.0, -1.0]
transition_covariance = np.eye(2)
observation_covariance = np.eye(2) + random_state.randn(2, 2) * 0.1
initial_state_mean = [5, -5]
initial_state_covariance = [[1, 0.1], [-0.1, 1]]

# sample from model
kf = KalmanFilter(
    transition_matrix, observation_matrix, transition_covariance,
    observation_covariance, transition_offset, observation_offset,
    initial_state_mean, initial_state_covariance,
    random_state=random_state
)
states, observations = kf.sample(
    n_timesteps=50,
    initial_state=initial_state_mean
)
print('this is for last observations \n\n\n')
print(type(observations))
print(observations[0,0])
print(observations.shape)
# estimate state with filtering and smoothing
filtered_state_estimates = kf.filter(observations)[0]
smoothed_state_estimates = kf.smooth(observations)[0]

# draw estimates
pl.figure()
lines_true = pl.plot(states, color='b')
lines_filt = pl.plot(filtered_state_estimates, color='r')
lines_smooth = pl.plot(smoothed_state_estimates, color='g')
pl.legend((lines_true[0], lines_filt[0], lines_smooth[0]),
          ('true', 'filt', 'smooth'),
          loc='lower right'
)
pl.show()
