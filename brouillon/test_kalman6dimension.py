#!/usr/local/bin/python3
import dateutil.parser as parser
import numpy as np
from numpy import dot, sum, tile, linalg
from numpy.linalg import inv
import sys
import os
import json
sys.path.insert(0, '/Users/NhatMinh/AVICEN/dummy_project/')

import config
import crash_detection
import pykalman

import numpy as np
import pylab as pl
from pykalman import KalmanFilter
import data_processing

def multi_plot():
    #_, axarr = plt.subplots(2, 2)
    pass

def get_data():
    level, time = 'moyenne', '9h45'
    level, time = 'faible', '9h28'
    #level, time = 'constant', '9h10'
    file = config.DICT_DATASET[level][time]
    path = os.path.join('data', file)
    data = open(path)
    raw_data = json.load(data)
    raw_data = data_processing.get_raw_data(raw_data)
    return raw_data
# specify parameters
random_state = np.random.RandomState(0)
delta_t = 0.001
transition_matrix = [[1, delta_t, delta_t, delta_t, delta_t, delta_t],
                     [delta_t, 1, delta_t, delta_t, delta_t, delta_t],
                     [delta_t, delta_t, 1, delta_t, delta_t, delta_t],
                     [delta_t, delta_t, delta_t, 1, delta_t, delta_t],
                     [delta_t, delta_t, delta_t, delta_t, 1,  delta_t],
                     [delta_t, delta_t, delta_t, delta_t, delta_t, 1]]
transition_offset = [delta_t, delta_t, delta_t, delta_t, delta_t, delta_t]
observation_matrix = np.eye(6) + random_state.randn(6, 6) * 0.01
observation_offset = [0.019777696141400954, -0.02717658649906955, 0.8597378069103433, .0, .0, .0]
transition_covariance = np.eye(6)
observation_covariance = np.eye(6) + random_state.randn(6, 6) * 0.01
#initial_state_mean = [5, -5]
initial_state_mean = [.0, .0, .0, .0, .0, .0]
initial_state_covariance = transition_matrix

# sample from model
kf = KalmanFilter(
    transition_matrix, observation_matrix, transition_covariance,
    observation_covariance, transition_offset, observation_offset,
    initial_state_mean, initial_state_covariance,
    random_state=random_state
)
observations = get_data()[:, :6]
# estimate state with filtering and smoothing
#filtered_state_estimates = kf.filter(observations)[0]
smoothed_state_estimates = kf.smooth(observations)[0]
print(smoothed_state_estimates.shape)
print('first estimated data rows:')
print(smoothed_state_estimates[:5,:])
print('some first observed data:')
print(observations[:5, :])
mean_obs_x = sum(observations[:, 0])/len(observations[:,0])
mean_est_x = sum(smoothed_state_estimates[:, 0])/len(smoothed_state_estimates[:,0])
print('means x observation = {}. mean x estimated = {}'.format(mean_obs_x, mean_est_x))
mean_obs_gyro_x = sum(observations[:, 3])/len(observations[:,3])
mean_est_gyro_x = sum(smoothed_state_estimates[:, 3])/len(smoothed_state_estimates[:,3])
print('means observation = {}. mean estimated = {}'.format(mean_obs_gyro_x, mean_est_gyro_x))
# draw estimates
_, axarr = pl.subplots(1, 3 )
lines_observedx = axarr[0].plot(observations[:,0], color='g')
lines_filtx = axarr[0].plot(smoothed_state_estimates[:,0], color='r')
pl.legend(( lines_filtx[0], lines_observedx[0]) ,  ( 'filt x', 'observation x'),
         loc='lower right')
lines_observedy = axarr[1].plot(observations[:,1], color='g')
lines_filty = axarr[1].plot(smoothed_state_estimates[:,1], color='r')
pl.legend(( lines_filty[0], lines_observedy[0]) ,  ( 'filt y', 'observation y'),
         loc='lower right')
lines_observedz = axarr[ 2].plot(observations[:,2], color='g')
lines_filtz = axarr[2].plot(smoothed_state_estimates[:,2], color='r')
pl.legend(( lines_filtz[0], lines_observedz[0]) ,  ( 'filt z', 'observation z'),
         loc='lower right')
pl.show()
