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
def main():
    level, time = 'moyenne', '9h45'
    level, time = 'faible', '9h28'
    file = config.DICT_DATASET[level][time]
    path = os.path.join('data', file)
    data = open(path)
    raw_data = json.load(data)
    raw_data = data_processing.get_raw_data(raw_data)
    return raw_data
# specify parameters
random_state = np.random.RandomState(0)
transition_matrix = [[1, 0.1], [0, 1]]
transition_offset = [-0.05, 0.05]
observation_matrix = np.eye(2) + random_state.randn(2, 2) * 0.1
observation_offset = [0.0, -0.0]
transition_covariance = np.eye(2)
observation_covariance = np.eye(2) + random_state.randn(2, 2) * 0.1
#initial_state_mean = [5, -5]
initial_state_mean = [.0, .0]
initial_state_covariance = [[1, 0.1], [-0.1, 1]]

# sample from model
kf = KalmanFilter(
    transition_matrix, observation_matrix, transition_covariance,
    observation_covariance, transition_offset, observation_offset,
    initial_state_mean, initial_state_covariance,
    random_state=random_state
)
observations = main()[:, [0,1]]

# estimate state with filtering and smoothing
filtered_state_estimates = kf.filter(observations)[0]
smoothed_state_estimates = kf.smooth(observations)[0]
print(smoothed_state_estimates.shape)
print(smoothed_state_estimates[:5,:])
print(observations[:5, :])

# draw estimates
pl.figure()
lines_filt = pl.plot(filtered_state_estimates[:,0], color='r')
lines_smooth = pl.plot(observations[:,0] +1, color='g')
#pl.legend(( lines_filt[0], lines_smooth[0]),
 #         ( 'filt', 'smooth'),
  #        loc='lower right'
#)
pl.show()