#!/usr/local/bin/python3
import numpy as np
SENSITIVITY_ACC = 16384 # for accelerometer of 16bit: 2^16, we suppose that our captor is use for
#range +/- 2g
SENSITIVITY_GYRO = 131 #see https://www.invensense.com/products/motion-tracking/6-axis/mpu-6050/
LEVEL0 = 0.8 #premier niveau de brusque freinage
LEVEL1  = 4 # normally is 4 but we modified to get some accident nofitication
# normally is 4 but we modified to get some accident nofitication
#LEVEL_DICT = {'sudden_braking':[1,2], 'mild_accident': [2, 6, 10, 14, 18],
 #             'medium_accident': [18, 23, 28, 34, 40], 'severe_accident': [40, 50, 60, 70]}
DURE_TIME_OF_AN_EVENT = 1.0
LEVEL_LIST = [1.5, 2.5, 6, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 1000] # 12 accidents levels and 1 sudden braking
LEVEL_LIST = [1.0, 2., 3., 4., 5., 20, 25, 30, 35, 40, 50, 60, 70, 1000]
SMALL_SCALE_LEVEL_LIST = [0.65, .88, .91,  1.5, 2.0, 3.0,  4.0, 6.0, 8.0, 10.0, 16.0, 32.0, 64.0, 1000]
LIST_TO_DETECT_ALL_STATE = [0, 0.15, 1.0, 2., 3., 4., 5., 20, 25, 30, 35, 40, 50, 60, 70, 1000]
PATH_OFFSET = 'data/constant_dataset.json' # path of constant data
#DATA_PATH = 'data/constant_dataset.txt'
#PREQUENCY = 50 #This is the frequecy of return data by sec. MUST BE CHANGED TO ADAPT OUR DEVICE
PREQUENCY = 50
OFFSET = np.asarray([0.019777696141400954, -0.02717658649906955, 0.8597378069103433]) #this value is obtain by running
# get_mean_from_constant_dataset()
ALPHA = 1/10 # this is a tuning parameter to combine Acc and Gyro for a more precise value of Acc
#this value 1/10 is not good because it is too sensitive to the Gyro
ALPHA = 7/10 #not good
#ALPHA = 19/20 #good value
ALPHA = 9/10
MOVEMENT_MSG = 'Vehicle is moving'
STOP_MSG = 'Vehicle is stopping'
FORWARD_MSG = 'Vehicle is forwarding'
BACKWARD_MSG = 'Vehicle is backwarding'
TIME_SLEEP = 0.5
#TIME_SLEEP = 1.0
SKIP_STEP = 15
STOP_LEVEL = 0.15
MAX_LENGTH = 8
MAX_LENGTH = 50
LEVEL_RANGE = [0, 1, 2, 3, 4]
STOP_LEVEL =  0.15
DICT_DATASET = {'constant':{'9h10': 'dataset_1541581764.json', '9h14' : 'dataset_1541582009.json',
                            '9h18':'dataset_1541582258.json', '10h03' : 'dataset_1541584927.json'},
                'faible':{'9h28': 'dataset_1541582884.json', '9h38': 'dataset_1541583464.json',
                          '9h39': 'dataset_1541583529.json'},
                'moyenne': {'9h42': 'dataset_1541583678.json', '9h44': 'dataset_1541583812.json',
                            '9h45' : 'dataset_1541583895.json'},
                'severe': {'9h51' : 'dataset_1541584249.json', '9h53' : 'dataset_1541584356.json',
                           '9h54' : 'dataset_1541584438.json'},
                'avance' : {'9h57' : 'dataset_1541584535.json', '10h00' : 'dataset_1541584762.json'},
                'avance_stop_recoule_a_fond' : {'10h05' : 'dataset_1541585108.json', '10h06': 'dataset_1541585181.json'}
                }
