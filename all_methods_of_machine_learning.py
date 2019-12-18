#!/usr/local/bin/python3
"""
There are parameters to be change to adapt our situation, like TIME, LEVEL etc
We can find all description of sensor here:
https://store.invensense.com/datasheets/invensense/MPU9250REV1.0.pdf
By this description, we must to divise our data to the sensitivity_acc and sensitivity_gyro
"""
import json
import numpy as np
import math
import time
import sys
import os
import dateutil.parser as parser
import config

DICT_DATASET = config.DICT_DATASET
PLOT = True
if PLOT:
    import matplotlib.pyplot as plt


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
LIST_TO_DETECT_ALL_STATE = [0, 0.15, 1.0, 2., 3., 4., 5., 20, 25, 30, 35, 40, 50, 60, 70, 1000]
PATH_OFFSET = 'data/constant_dataset.json' # path of constant data
DATA_PATH = 'data/document.json'
DATA_PATH = 'data/accelerometer_gyroscope_9_crashes.json'
DATA_PATH = 'data/dataset_choc_faible_1.json'
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
ALPHA = 1
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
MAX_LENGTH = 100
LEVEL_RANGE = [0, 1, 2, 3, 4]

def get_raw_data(level, time):
    """
    :param level: 'faible', 'moyenne'...
    :param time: '9h10', '10h00', ...
    :return: raw_data
    """

    file = DICT_DATASET[level][time]
    path = os.path.join('data', file)
    data = open(path)
    data = json.load(data)
    #print('Opening data file at {}'.format(path))
    index_count = 0
    infos = str(level) + ' at ' + str(time)
    for index, line in enumerate(data):
        date_time = line['datetime']
        date_time = parser.parse(date_time)
        if index == 0:
            start_time = date_time
        acc = line['accelerometer'][0]
        acc_x = float(acc['x'])/SENSITIVITY_ACC
        acc_y = float(acc['y'])/SENSITIVITY_ACC
        acc_z = float(acc['z'])/SENSITIVITY_ACC
        gyro = line['gyroscope'][0]
        gyro_x = float(gyro['x'])/SENSITIVITY_GYRO
        gyro_y = float(gyro['y'])/SENSITIVITY_GYRO
        gyro_z = float(gyro['z'])/SENSITIVITY_GYRO
        gyro_x = degree_to_radian(gyro_x)
        gyro_y = degree_to_radian(gyro_y)
        gyro_z = degree_to_radian(gyro_z)
       # new_row = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, datetime]
        new_row = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, date_time]
        if index_count == 0:
            data_processed = np.asanyarray(new_row)
        else:
            data_processed = np.vstack([data_processed, new_row])
        index_count += 1
    #print('Get numpy array data set: infos {}'.format(infos))
    time_ = (date_time - start_time).total_seconds()
   # print('this data set dure for {:.2f} seconds'.format(time_))
   # print('Frequency is {:.2f} Hz: numbers of points data/total seconds'.format(index/time_))
    return data_processed, infos

def simple_data_processing(raw_data):
    """
    :param raw_data: numpy array where there are acc, gyro, time
    :return: raw_data with 4 axes acc and time gyro deleted and  reduced offset in three first axe
    """
    raw_data = raw_data[:, [0, 1, 2, 6]]
    raw_data[:,:3] = raw_data[:, :3] - OFFSET
    return raw_data

def test_data(level, time):

    file = DICT_DATASET[level][time]
    path = os.path.join('data', file)
    data = open(path)
    data = json.load(data)
    for index, line in enumerate(data):
        datetime_ = line['datetime']
        date = parser.parse(datetime_)
        print(date)
        print(type(date))
        if index >0:
            print('difference between date ', (date - last_date).total_seconds())
        numpyarr = np.asarray(date)
        print('numpy ', numpyarr)
        last_date = date
        if index >5:
            break

def euclid_distance(x,y):
    """
    :param x: np.array vector
    :param y: np.array vector of the same length of x
    :return: distance between x and y
    """
    return np.sqrt(sum(np.square(x-y)))

def euclid_norm(x):

    return np.sqrt(sum(np.square(x)))


def degree_to_radian(x):

    return x * math.pi / 180


def accident_level(time_, value_g):
    """
    :param value_g:
    :return: message of sudden braking and accident level
    """
    if value_g < LEVEL_LIST[0]:
        return None
    dictionary = {'value': "%.2f" % value_g, 'datetime': time_}
    for index in range(len(LEVEL_LIST)-1):
        if value_g >= LEVEL_LIST[index] and value_g < LEVEL_LIST[index +1]:
            dictionary['level'] = index
            if index == 0:
                dictionary['message'] = 'SUDDEN BRAKING'
            if index>0 and index <5:
                dictionary['message'] = 'MILD ACCIDENT'
            if index >=5 and index <9:
                dictionary['message'] = 'MEDIUM ACCIDENT'
            if index >= 9:
                dictionary['message'] = 'SEVERE ACCIDENT'
    return dictionary

def filter_event(last_event, this_event):
    """
    :param last_event: infos for last event: message, level, value...
    :param this_event: infos for this event
    :return: event more important with bigger value.
    """
    if not last_event:
        return this_event
    if (this_event['datetime'] - last_event['datetime']).total_seconds() > DURE_TIME_OF_AN_EVENT:
        return [last_event, this_event]
    if this_event['value'] >= last_event['value']:
        return this_event
    else:
        return last_event

def print_last_and_return_this_event(last_event, this_event):

    all_event = filter_event(last_event, this_event)
    if len(all_event) == 2:
        message = all_event[0]['message']
        level  = all_event[0]['level']
        if level != 0:
            time_ = all_event[0]['datetime'].replace(microsecond=0)
            message = str(message) + ' level ' + str(level) + '/12' + ' at ' + str(time_)
        print(message)
        full_message = message + ' value =  {}'.format(all_event[0]['value'])
        return this_event, full_message
    else:
       return all_event, None # which have only one event by filter

def get_mean_from_constant_dataset():
    """
    :param path: path of constant dataset
    :return: np.array of mean of this dataset from x, y, z accelerometers
    """
    print('GET OFFSET: ')
    raw_data, infos = get_raw_data('constant', '10h03')
    #print('Opening  {} DATASET to get offset'.format(infos))
    return np.sum(raw_data[:,:3], axis = 0)/len(raw_data[:, 0])


def get_accelerometer_and_time(json_data):
    """
    :return: Accelometer  and gyroscope of this data in numpy array
    """
    index_count = 0
    for line in json_data:
        datetime = line['datetime']
        acc = line['accelerometer'][0]
        acc_x = float(acc['x'])/SENSITIVITY_ACC
        acc_y = float(acc['y'])/SENSITIVITY_ACC
        acc_z = float(acc['z'])/SENSITIVITY_ACC
        gyro = line['gyroscope'][0]
        gyro_x = float(gyro['x'])/SENSITIVITY_GYRO
        gyro_y = float(gyro['y'])/SENSITIVITY_GYRO
        gyro_z = float(gyro['z'])/SENSITIVITY_GYRO
        new_row = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, datetime]
        if index_count == 0:
            data_processed = np.asanyarray(new_row)
        else:
            data_processed = np.vstack([data_processed, new_row])
        index_count += 1
    return data_processed

def reduce_noise_by_average(estimated_acc, M):

    """
    :param estimated_acc: is 3D array data
    :param M: number of ligns to average
    :return: numpy array with noise reduced
    """
    estimated_acc_average = 0
    estimated_acc_average_i = 0
    for index, line in enumerate(estimated_acc):
        if index + M >= len(estimated_acc):
            break
        for i in range(M):
            estimated_acc_average_i += estimated_acc[i+index, :]/M
        if index == 0:
            estimated_acc_average = estimated_acc_average_i
        else:
            estimated_acc_average = np.vstack([estimated_acc_average, estimated_acc_average_i])
        estimated_acc_average_i = 0
    return estimated_acc_average

def reduce_noise_in_data_by_using_only_x_y(raw_data):
    """
    :param alpha: Tuning parameter
    :return: numpy array of  x,y,z estimated
    all  calculations are from http://www.starlino.com/imu_guide.html
    """
    estimated_acc = raw_data[:,:3]
    zero_column = np.zeros((len(estimated_acc[:,0]),1))
    estimated_acc[:,2] = zero_column[:,0]
    print('line 0 of data {}'.format(estimated_acc[0,:]))
    print('line 1 of data {}'.format(estimated_acc[ 1, :]))
    return estimated_acc

def crash_detection_from_raw_data(raw_data):
    """
    In one second, there is max an accident
    :param raw_data: { [acc_x, ] }
    :return: All messages about accident
    """
    raw_data = simple_data_processing(raw_data)
    last_event = None
    for  line in raw_data:
            value_g = np.sqrt(sum(np.square(line[:3])))
            time_ = line[-1]
            this_event = accident_level(time_, value_g)
            if this_event:
                last_event, _ = print_last_and_return_this_event(last_event, this_event)


def get_message(message):
    """
    :param message
    :return : new message
    """
    if '.'*30  in message:
        message = message.strip('.')
    else:
        message = message + '.'
    return message


def mouvement_detection(raw_data):

    LEVEL_STOP = 0.15
    #last_time = datetime
    last_status, this_status = 0, 0 # the first state is consider that
    last_message = 'vehicle in stop'
    number_strip = 1 # nombre de trajets
    count_time_stop = 0
    boolean = False
    raw_data = simple_data_processing(raw_data)
    for line in raw_data:
        if index > 0:
            value_g =  np.sqrt(sum(np.square(line[:3])))
            this_time = line[-1]
            if value_g >= LEVEL_STOP and (this_time -last_time).all_seconds() > DURE_TIME_OF_AN_EVENT:
                this_status = 1
                message_dict = get_message(last_message_dict, last_status, this_status)
                last_message = message_dict
                last_time = this_time
                print(message, end = '\r')
                time.sleep(.5)
            if value_g < LEVEL_STOP and (this_time -last_time).all_seconds() > DURE_TIME_OF_AN_EVENT:
                this_status = 0
                if last_status == 1:
                    boolean = 1 # pour commencer à calculer le temps d'arrêt
                    count_time_stop = 0
                if boolean:
                    count_time_stop += 1
                if count_time_stop > DU:# si le vehicule s'arrête plus d'une second, ça va
                    #compter un vrai arrêt, donc,
                    # ajouter un trajet de plus au nombre total  des trajets
                    number_strip += 1
                    boolean = False
                    count_time_stop = 0
                message = get_message(last_message, last_status, this_status)
                last_message = message
                print(message, end = '\r')
                time.sleep(.5)
                last_index_print = index
            sys.stdout.write("\033[K")
        last_status = this_status
        last_line = line
    print('Number of trips is {}'.format(number_strip))
    #return number_strip #nombre de trajets

def is_move(g_value):

    """
    :param g_value:
    :return: False if in stop, True if in moving
    """
    if g_value >= STOP_LEVEL:
        return True
    else:
        return False

def message_by_stuation(is_move):

    if is_move:
        return MOVEMENT_MSG
    else:
        return STOP_MSG

def nor_operation(boolean1, boolean2):
    """
    :param boolean1:
    :param boolean2:
    :return: True only if only and only boolean is True, else, return False
    """
    if boolean1 == boolean2:
        return False
    else:
        return True

def movement_stop_detection(raw_data):

    raw_data = simple_data_processing(raw_data)
    num_event = 0
    for  index, line in enumerate(raw_data):
        g_value = np.sqrt(sum(np.square(line[:3])))
        time_ = line[-1]
        if index == 0:
            time_start_event = time_
            is_moving = is_move(g_value)
            message = message_by_stuation(is_moving)
            tuning_step = 0
            print(message, end = '\r')
            time.sleep(TIME_SLEEP)
        else:
            if nor_operation(is_moving, is_move(g_value)):
                if tuning_step > SKIP_STEP:
                    is_moving = is_move(g_value)
                    message = message_by_stuation(is_moving)
                    print(message, end = '\r')
                    time.sleep(TIME_SLEEP)
                    time_start_event = time_
                    tuning_step = 0
                    num_event += 1
                else:
                    tuning_step += 1
                    if (time_ - time_start_event).total_seconds() > 1:
                        message = get_message(message)
                        print(message, end = '\r')
                        #print(message)
                        time.sleep(TIME_SLEEP)
                        time_start_event = time_
            sys.stdout.write("\033[K")
    print('-'*50)
    print('total of change of states: {} times'.format(num_event))

def find_state(g_value):

    if g_value >= STOP_LEVEL:
        return 1
    elif -g_value >= STOP_LEVEL:
        return -1
    else:
        return 0

def message_for_three_situation(state):

    if state == 1:
        return FORWARD_MSG
    if state == 0:
        return STOP_MSG
    else:
        return BACKWARD_MSG

def is_tuning_state(last_state, this_state):
    """
    :param last_state:
    :param this_state:
    :return: True if this_state is not last_state else, return False
    """
    if last_state == this_state:
        return False
    else:
        return True

def forward_backward_stop_detection(raw_data):

    raw_data = simple_data_processing(raw_data)
    num_event = 0
    for  index, line in enumerate(raw_data):
        if sum(line[:3]) >= 0:
            sign_g = 1
        else:
            sign_g = -1
        g_value = np.sqrt(sum(np.square(line[:3]))) * sign_g
        time_ = line[-1]
        if index == 0:
            time_start_event = time_
            state = find_state(g_value)
            message = message_for_three_situation(state)
            tuning_step = 0
            save_state = state
            print(message, end = '\r')
            time.sleep(TIME_SLEEP)
        else:
            if is_tuning_state(state, find_state(g_value)):
                if tuning_step > SKIP_STEP and save_state == find_state(g_value):
                    state = find_state(g_value)
                    message = message_for_three_situation(state)
                    print(message, end = '\r')
                    time.sleep(TIME_SLEEP)
                    time_start_event = time_
                    tuning_step = 0
                    num_event += 1
                else:
                    if save_state == find_state(g_value):
                        tuning_step += 1
                    else:
                        save_state = find_state(g_value)
                        tuning_step = 0
            else:
                    if (time_ - time_start_event).total_seconds() > 1:
                        message = get_message(message)
                        print(message, end = '\r')
                        #print(message)
                        time.sleep(TIME_SLEEP)
                        time_start_event = time_
            sys.stdout.write("\033[K")
    print('-'*50)
    print('change states for {} times'.format(num_event))


def find_all_state(g_value, time_):
    """
    :param g_value:
    :return: dict_of_all_message
    """
    abs_g = abs(g_value)
    dictionary = {'value': "%.2f" % g_value, 'datetime': time_}
    time_print = time_.replace(microsecond=0)
    for index in range(len(LIST_TO_DETECT_ALL_STATE)-1):
        if abs_g >= LIST_TO_DETECT_ALL_STATE[index] and abs_g < LIST_TO_DETECT_ALL_STATE[index +1]:
            if index == 0:
                raw_index = 0
                dictionary['message'] = STOP_MSG
            if index == 1:
                if g_value < 0:
                    dictionary['message'] = BACKWARD_MSG
                    raw_index = 1
                else:
                    dictionary['message'] = FORWARD_MSG
                    raw_index = 2
            if index == 2:
                dictionary['message'] = 'SUDDEN BRAKING at ' + str(time_print)
                raw_index = 3
            if index > 2:
                raw_index = 4
            if index>2 and index <6:
                dictionary['message'] = 'MILD ACCIDENT level '+ str(index +2) + '/12 at ' + str(time_print)
            if index >=6 and index <11:
                dictionary['message'] = 'MEDIUM ACCIDENT level'+ str(index +2) + '/12 at ' + str(time_print)
            if index >= 11:
                dictionary['message'] = 'SEVERE ACCIDENT level' + str(index +2) + '/12 at ' + str(time_print)
    dictionary['level'] = raw_index
    return dictionary

def filter_all_event(list_of_state, level_printing, this_level):
    """
    :param: list of some last state
    :return: True, state of change else, return False, None
    """
    if len(list_of_state) < MAX_LENGTH or (level_printing == this_level):
        return False, None
    for range in LEVEL_RANGE:
        indices = [i for i, x in enumerate(list_of_state) if x == range]
        if len(indices)/len(list_of_state) >= 0.6:
            return True, range
    return False, None

def get_new_list_of_state(list_of_state, raw_level):
    """
    :param list_of_state: list of state of value 0, 1, 2, 3, 4, which match with STOP BACKWARD FORWARD SUDDEN_BRAKING,
    and ACCIDENT
    :param raw_level: one of above level
    :return: new list
    """
    list_of_state.append(raw_level)
    if len(list_of_state) <= MAX_LENGTH:
        return list_of_state
    else:
        return list_of_state[1:]


def all_detections(raw_data):
    """
    :param raw_data: 7 axes raw data
    :return: print all message ebout state of  vehicle
    """
    start = time.time()
    print('-'*50)
    raw_data = simple_data_processing(raw_data)
    list_of_state = []
    for index, line in enumerate(raw_data):
        time_start_this_loop = time.time()
        if sum(line[:3]) >= 0:
            sign_g = 1
        else:
            sign_g = -1
        g_value = euclid_norm(line[:3]) * sign_g
        this_time = line[-1]
        this_state = find_all_state(g_value, this_time)
        this_level = this_state['level']
        #print('this state: ', this_state)
        list_of_state = get_new_list_of_state(list_of_state, this_state['level'])
        #print('list state', list_of_state)
        if index == 0:
            message = this_state['message']
            print(message, end = '\r')
            #print(message)
            level_printting = this_state['level']
            time_start_event = this_time
            last_state = this_state
        else:
            is_tuning, _ = filter_all_event(list_of_state, level_printting, this_level)
            if is_tuning:
                list_of_state = [] # delete all history of states
                message = this_state['message']
                last_state = this_state
                level_printting = this_state['level']
                if this_state['level'] > 2:
                    print(message)
                else:
                   # print(message, end = '\r')
                    print(message)
                time_start_event = this_time
                time_sleep = TIME_SLEEP - (time.time() - time_start_this_loop)
                #print('time sleep is {:.2f}', time_sleep)
                time.sleep(time_sleep)

            else:
                if (this_time - time_start_event).total_seconds() > 1:
                    message = last_state['message']
                    message = get_message(message)
                    last_state['message'] = message
                    #print(message, end = '\r')
                    #print(message)
                    #print(message)
                    time_sleep = TIME_SLEEP - (time.time() - time_start_this_loop)

                    #print('time sleep is {:.2f}', time_sleep)
                    #time.sleep(time_sleep)
                    time_start_event = this_time
            #sys.stdout.write("\033[K")
    print('-' * 50)
    print('all run time is', time.time() - start)

def all_detections_with_time_(raw_data):
    """
        :param raw_data: 7 axes raw data
        :return: print all message ebout state of  vehicle
        """
    start = time.time()
    print('-' * 50)
    raw_data = simple_data_processing(raw_data)
    list_of_state = []
    for index, line in enumerate(raw_data):
        time_start_this_loop = time.time()
        if sum(line[:3]) >= 0:
            sign_g = 1
        else:
            sign_g = -1
        g_value = euclid_norm(line[:3]) * sign_g
        this_time = line[-1]
        this_state = find_all_state(g_value, this_time)
        this_level = this_state['level']
        # print('this state: ', this_state)
        list_of_state = get_new_list_of_state(list_of_state, this_state['level'])
        # print('list state', list_of_state)
        if index == 0:
            message = this_state['message']
            print(message, end = '\r')
            # print(message)
            level_printting = this_state['level']
            time_start_event = this_time
            last_state = this_state
        else:
            is_tuning, _ = filter_all_event(list_of_state, level_printting, this_level)
            if is_tuning:
                message = this_state['message']
                last_state = this_state
                level_printting = this_state['level']
                if this_state['level'] > 2:
                    print(message)
                else:
                    print(message, end = '\r')
                    # print(message)
                time_start_event = this_time
                time_sleep = TIME_SLEEP - (time.time() - time_start_this_loop)
                # print('time sleep is {:.2f}', time_sleep)
                time.sleep(time_sleep)

            else:
                if (this_time - time_start_event).total_seconds() > 1:
                    message = last_state['message']
                    message = get_message(message)
                    last_state['message'] = message
                    print(message, end = '\r')
                    # print(message)
                    # print(message)
                    time_sleep = TIME_SLEEP - (time.time() - time_start_this_loop)

                    # print('time sleep is {:.2f}', time_sleep)
                    time.sleep(time_sleep)
                    time_start_event = this_time
            # sys.stdout.write("\033[K")
    print('-' * 50)
    print('all run time is', time.time() - start)


def plot_all_g(data_processed):

    #import matplotlib.pyplot as plt
    print('length of data set is {}'.format(len(data_processed)))
    list_y = []
    for index, line in enumerate(data_processed):
        if index > 0:
            value_g = PREQUENCY * np.sqrt(sum(np.square(line )))
            list_y.append(value_g)
        last_line = line
    list_x = range(len(list_y))
    print('max = {} and min = {}'.format(max(list_y), min(list_y)))
    plt.plot(list_x, list_y)
    plt.show()

def update_kalman_filter(last_value, this_raw_value, last_p, R = 0.1):

    this_k = last_p/(last_p + R)
    this_value = (1 - this_k)*last_value + this_k * this_raw_value
    this_p = (1-this_k) * last_p
    return this_value, this_p

def processing_data_by_agreated_and_using_kalman_filter(raw_data, M = 1):
    """
    :param raw_data: numpy array, for now, three axes
    :param M: integer, number of line to agreate
    using simple Kalman filter
    :return: data_processed
    """
    if M > 1:
        N = len(raw_data[:,0]) // M
        for index in range(N):
            this_row = raw_data[index*M, :]
            for i in range(M-1):
                this_row += raw_data[index*M+i+1, :] / M
            if index == 0:
                data_processed = np.asarray(this_row)
            else:
                data_processed = np.vstack([data_processed, this_row])
    else:
        data_processed = raw_data
    last_x = data_processed[0, 0]
    last_y = data_processed[0, 1]
    last_z = data_processed[0, 2]
    last_px, last_py, last_pz = 1, 1, 1
    for index in range(len(data_processed)):
        if index == 0:
            data_processed_by_kalman = np.asarray([last_x, last_y, last_z])
        else:
            raw_value_x = data_processed[index, 0]
            raw_value_y = data_processed[index, 1]
            raw_value_z = data_processed[index, 2]
            last_x, last_px = update_kalman_filter(last_x, raw_value_x, last_px )
            last_y, last_py = update_kalman_filter(last_y, raw_value_y, last_py)
            last_z, last_pz = update_kalman_filter(last_z, raw_value_z, last_pz)
            new_row = [last_x, last_y, last_z]
            data_processed_by_kalman = np.vstack([data_processed_by_kalman, new_row])
    return data_processed_by_kalman

def processing_data_by_agreated(raw_data, M):
    """
    :param raw_data: numpy array of three axes
    :param M: integer, number of line to agreate
    :return: numpy array data processed
    """
    if M == 1:
        return raw_data
    N = len(raw_data[:,0])//M
    for index in range(N):
        this_row = raw_data[index * M, :3]
        for i in range(M-1):
            this_row += raw_data[index * M + i + 1, :3]/M
        if index == 0:
            data_processed = np.asarray(this_row)
        else:
            data_processed = np.vstack([data_processed, this_row])
    return data_processed


def processing_data_by_agreated_with_time(raw_data, M):
    """
    :param raw_data: numpy array of 7 axes
    :param M: integer, number of line to agreate
    :return: numpy array data processed
    """
    raw_data = raw_data[:, [0, 1, 2, 6]]
    #print(raw_data[0,:])
    if M == 1:
        return raw_data
    N = len(raw_data[:,0])//M
    for index in range(N):
        this_row = raw_data[index * M, :3]
        for i in range(M-1):
            this_row += raw_data[index * M + i + 1, :3]/M
        this_row = np.append(this_row, raw_data[index*M, 3])
        if index == 0:
            data_processed = np.asarray(this_row)
           # print(data_processed)
        else:
            data_processed = np.vstack([data_processed, this_row])
    return data_processed


def plot_g_value_list(list):

    print('plotting...')
    print('value of g, max {} and min {}'.format(max(list), min(list)))
    list_x = range(len(list))
    plt.plot(list_x, list)
    plt.show()


def some_test_code():

    pass

def get_list_of_g(raw_data):
    """
    :param data: is 3D accelerometer clean data, with a given frequen
    M: is number of lines to reduce
    :return: numpy array data of acceleration or decceleration
    """
    list_ = []
    for index, line in enumerate(raw_data):
        vector_g = sum(line[:3])
        if vector_g >= 0:
            sign_g = 1
        else:
            sign_g = -1
        value_g = np.sqrt(sum(np.square(line[:3])))*sign_g
        list_.append(value_g)
    return list_


def analyse_constant_dataset(M):
    """
    :param M: integer from 1 to 4 say what constant data we use
    :return:
    """
    if M == 0:
        time = '9h10'
    elif M == 1:
        time = '9h14'
    elif M == 2:
        time = '9h18'
    else:
        time = '10h03'
    raw_data, infos = get_raw_data('constant', time)
    print('get {} from dataset')
    raw_data = raw_data[:, :3]

def multi_plots(level, time):

    raw_data, infos = get_raw_data(level, time)
    raw_data = raw_data[:, :3]
    raw_data1 = raw_data
    raw_data2 = processing_data_by_agreated(raw_data, 1)
    raw_data3 = raw_data - OFFSET
    raw_data4 = processing_data_by_agreated_and_using_kalman_filter(raw_data, 1) - OFFSET
    array_raw_data = [[raw_data1, raw_data2], [raw_data3, raw_data4]]
    _, axarr = plt.subplots(2, 2)
    for i, raw_data_list in enumerate(array_raw_data):
        for j, raw_data in enumerate(raw_data_list):
            list_x = range(len(raw_data[:, 0]))
            list_g = []
            data = raw_data
            for line in data:
                list_g.append(np.sqrt(sum(np.square(line))))
            #axarr[i, j].plot(list_x, data[:, 0], color='purple')
            #axarr[i, j].plot(list_x, data[:, 1], color='green')
            axarr[i, j].plot(list_x, data[:, 2], color='yellow')
            #axarr[i, j].plot(list_x, list_g, color='red')
            #axarr[i, j].legend(('x', 'y', 'z', 'g'),
             #          loc='lower right')
    #plt.title(infos, loc = 'center')
    plt.show()

def write_detection_of_movement_to_list(raw_data):
    """
    :param raw_data: is np array 7 axes
    :return: np array 1 axes of value -1, 0, -1
    """
    raw_data = simple_data_processing(raw_data)
    list = []
    for index, line in enumerate(raw_data):
        if sum(line[:3]) >= 0:
            sign_g = 1
        else:
            sign_g = -1
        g_value = np.sqrt(sum(np.square(line[:3]))) * sign_g
        state = find_state(g_value)
        if index == 0:
            list.append(state)
            tuning_step = 0
            last_state = state
            save_state = state
        else:
            if is_tuning_state(state, last_state):
                if tuning_step > SKIP_STEP:
                    tuning_step = 0
                    list.append(state)
                    last_state = state
                else:
                    list.append(list[-1])
                    if save_state == state:
                        tuning_step += 1
                    else:
                        save_state = state
                        tuning_step = 0
            else:
                list.append(list[-1])
    return np.asarray(list)



def plots_of_movement(level, time):

    raw_data0, infos = get_raw_data(level, time)
    raw_data1 = write_detection_of_movement_to_list(raw_data0)
    list_x = range(len(raw_data1))
    plt.plot(list_x, raw_data1)
    plt.show()

def save_all_result_to_file():

    path = 'data/result.txt'
    #f  = open(path, 'w') #to delete all existing data in path
    f = open(path, 'a') #to append new data
    f.write('METHOD: DELETE OFFSET AND USE KALMAN FILTER with m = 1 \n')
    f.write('rawdata --> reduce offset --> kalman filter \n')
    f.write('List level: ' + str(LEVEL_LIST) + '\n\n')
    for level in DICT_DATASET.keys():
        for time in DICT_DATASET[level].keys():
            print(level, time)
            raw_data, infos = get_raw_data(level, time)
            f.write(infos + '\n')
            raw_data = raw_data[:, [0, 1, 2, 6]]
            raw_data[:,:3] = raw_data[:,:3] - OFFSET  # reduce bias of earth's gravity
            raw_data[:, :3] =  processing_data_by_agreated_and_using_kalman_filter(raw_data[:, :3], M = 1)
            last_event = None
            for  line in raw_data:
                value_g = np.sqrt(sum(np.square(line[:3])))
                time = line[-1]
                this_event = accident_level(time, value_g)
                if this_event:
                    last_event, message = print_last_and_return_this_event(last_event, this_event)
                    if message:
                        f.write(message + '\n')
    f.write('-'*80 + '\n')
    f.close()

def get_first_level_from_data(level = 1, agreate = 5):
    """

    :return: all level float detection by this function to use lately:
    Ideally, it must return for each data set, not constant 3 accident
    level = 1: must detect 3X3X3 = 27 accidents
    level = 2: must detect 3X3 = 9 accidents
    level = 3: must detect 3 accidents
    """
    path = 'data/result.txt'
    f = open(path, 'a') #to append new data
    f.write('METHOD: DELETE OFFSET AND USE AGRERATE WITH M =' + str(agreate) + '\n\n')
    dictionnary = {'faible':{'9h28': 'dataset_1541582884.json', '9h38': 'dataset_1541583464.json',
                          '9h39': 'dataset_1541583529.json'},
                'moyenne': {'9h42': 'dataset_1541583678.json', '9h44': 'dataset_1541583812.json',
                            '9h45' : 'dataset_1541583895.json'},
                'severe': {'9h51' : 'dataset_1541584249.json', '9h53' : 'dataset_1541584356.json',
                           '9h54' : 'dataset_1541584438.json'}}
    level1 = 2.0
    level_to_fit_data = level1
    all_lower_level1 = []
    all_upper_level1 = []
    dictionnary_to_write_in_file = {}
    while True:
        total_accident = 0
        last_event = None
        for level in dictionnary.keys():
          for time in dictionnary[level].keys():
            last_event = None
            #print(level, time)
            raw_data, infos = get_raw_data(level, time)
            raw_data[:,:3] = raw_data[:,:3] - OFFSET  # reduce bias of earth's gravity
            raw_data =  processing_data_by_agreated_with_time(raw_data, agreate)
            for  line in raw_data:
                value_g = np.sqrt(sum(np.square(line[:3])))
                time = line[-1]
                this_event = accident_level_to_fit_level(level_to_fit_data, time, value_g)
                if this_event:
                    last_event, number = filter_accident_to_fit_level(last_event, this_event)
                    total_accident += number
        if  26<= total_accident <= 29:
            print('total accident is {}. level to find is {:.2f}'
                  .format(total_accident, level_to_fit_data))
            break
        else:
            print('total accident detected for this loop is {}, with level {}'
                  .format(total_accident, level_to_fit_data))
            dictionnary_to_write_in_file[level_to_fit_data] = total_accident
            if total_accident <26: #this level is too big
                all_upper_level1.append(level_to_fit_data)
                if not all_lower_level1:
                    level_to_fit_data /= 2
                else:
                    level_to_fit_data = (max(all_lower_level1) + min(all_upper_level1))/2
            else: # In this case, level is to small, we must inscrase it
                all_lower_level1.append(level_to_fit_data)
                if not all_lower_level1:
                    level_to_fit_data *= 2
                else:
                    level_to_fit_data = (max(all_lower_level1) + min(all_upper_level1))/2

    f.write('-'*80 + '\n')
    f.write('good level 1 is {.:2f} \n'.format(level_to_fit_data))
    f.write('-'*80 + '\n')
    f.close()
    return level_to_fit_data

def get_second_level_from_data(level_medium = 2, agreate = 5):
    """

    :return: all level float detection by this function to use lately:
    Ideally, it must return for each data set, not constant 3 accident
    level = 1: must detect 3X3X3 = 27 accidents
    level = 2: must detect 2X3X3 = 18 accidents
    level = 3: must detect 3X3 accidents
    """
    path = 'data/result.txt'
    f = open(path, 'a') #to append new data
    f.write('METHOD: DELETE OFFSET AND USE AGRERATE WITH M =' + str(agreate) + '\n\n')
    f.write('detect level 2: get all medium accidents\n')
    dictionnary = {
                'moyenne': {'9h42': 'dataset_1541583678.json', '9h44': 'dataset_1541583812.json',
                            '9h45' : 'dataset_1541583895.json'},
                'severe': {'9h51' : 'dataset_1541584249.json', '9h53' : 'dataset_1541584356.json',
                           '9h54' : 'dataset_1541584438.json'}}
    level_to_fit_data = level_medium
    all_lower_level1 = []
    all_upper_level1 = []
    dictionnary_to_write_in_file = {}
    while True:
        total_accident = 0
        last_event = None
        for level in dictionnary.keys():
          for time in dictionnary[level].keys():
            last_event = None
            #print(level, time)
            raw_data, infos = get_raw_data(level, time)
            raw_data[:,:3] = raw_data[:,:3] - OFFSET  # reduce bias of earth's gravity
            raw_data =  processing_data_by_agreated_with_time(raw_data, agreate)
            for  line in raw_data:
                value_g = np.sqrt(sum(np.square(line[:3])))
                time = line[-1]
                this_event = accident_level_to_fit_level(level_to_fit_data, time, value_g)
                if this_event:
                    last_event, number = filter_accident_to_fit_level(last_event, this_event)
                    total_accident += number
        if  17<= total_accident <= 19:
            print('total accident is {}. level to find is {:.2f}'
                  .format(total_accident, level_to_fit_data))
            break
        else:
            print('total accident detected for this loop is {}, with level {}'
                  .format(total_accident, level_to_fit_data))
            dictionnary_to_write_in_file[level_to_fit_data] = total_accident
            if total_accident <17: #this level is too big
                all_upper_level1.append(level_to_fit_data)
                if not all_lower_level1:
                    level_to_fit_data /= 2
                else:
                    level_to_fit_data = (max(all_lower_level1) + min(all_upper_level1))/2
            else: # In this case, level is to small, we must inscrase it
                all_lower_level1.append(level_to_fit_data)
                if not all_upper_level1:
                    level_to_fit_data *= 2
                else:
                    level_to_fit_data = (max(all_lower_level1) + min(all_upper_level1))/2

    f.write('-'*80 + '\n')
    f.write('good level 1 is {.:2f} \n'.format(level_to_fit_data))
    f.write('-'*80 + '\n')
    f.close()
    return level_to_fit_data


def get_third_level_from_data(level_severe = 3.0, agreate = 5):
    """

    :return: all level float detection by this function to use lately:
    Ideally, it must return for each data set, not constant 3 accident
    level = 1: must detect 3X3X3 = 27 accidents
    level = 2: must detect 3X3 = 9 accidents
    level = 3: must detect 3 accidents
    """
    path = 'data/result.txt'
    f = open(path, 'a') #to append new data
    f.write('METHOD: DELETE OFFSET AND USE AGRERATE WITH M =' + str(agreate) + '\n\n')
    f.write('detect level 2: get all medium accidents\n')
    dictionnary = {
                'severe': {'9h51' : 'dataset_1541584249.json', '9h53' : 'dataset_1541584356.json',
                           '9h54' : 'dataset_1541584438.json'}}
    level_to_fit_data = level_severe
    all_lower_level1 = []
    all_upper_level1 = []
    dictionnary_to_write_in_file = {}
    while True:
        total_accident = 0
        last_event = None
        for level in dictionnary.keys():
          for time in dictionnary[level].keys():
            last_event = None
            #print(level, time)
            raw_data, infos = get_raw_data(level, time)
            raw_data[:,:3] = raw_data[:,:3] - OFFSET  # reduce bias of earth's gravity
            raw_data =  processing_data_by_agreated_with_time(raw_data, agreate)
            for  line in raw_data:
                value_g = np.sqrt(sum(np.square(line[:3])))
                time = line[-1]
                this_event = accident_level_to_fit_level(level_to_fit_data, time, value_g)
                if this_event:
                    last_event, number = filter_accident_to_fit_level(last_event, this_event)
                    total_accident += number
        if  7<= total_accident <= 9:
            print('total accident is {}. level to find is {:.2f}'
                  .format(total_accident, level_to_fit_data))
            break
        else:
            print('total accident detected for this loop is {}, with level {}'
                  .format(total_accident, level_to_fit_data))
            dictionnary_to_write_in_file[level_to_fit_data] = total_accident
            if total_accident <7: #this level is too big
                all_upper_level1.append(level_to_fit_data)
                if not all_lower_level1:
                    level_to_fit_data /= 2
                else:
                    level_to_fit_data = (max(all_lower_level1) + min(all_upper_level1))/2
            else: # In this case, level is to small, we must inscrase it
                all_lower_level1.append(level_to_fit_data)
                if not all_upper_level1:
                    level_to_fit_data *= 2
                else:
                    level_to_fit_data = (max(all_lower_level1) + min(all_upper_level1))/2

    f.write('-'*80 + '\n')
    f.write('good level 1 is {.:2f} \n'.format(level_to_fit_data))
    f.write('-'*80 + '\n')
    f.close()
    return level_to_fit_data



def accident_level_to_fit_level(level, time_, value_g):
    """
    :param value_g:
    :return: message of sudden braking and accident level
    """
    if value_g < level:
        return None
    dictionary = {'value': "%.2f" % value_g, 'datetime': time_}
    return dictionary

def filter_accident_to_fit_level(last_event, this_event):
    """
    :param this_event: dict of value and time
    :param last_event: dict of value and time, mayby None
    :return: event as the same format of these precedent dict, and number of event to save: 0 or 1
    """
    if not last_event:
        return this_event, 0
    this_time = this_event['datetime']
    last_time = last_event['datetime']
    this_value = this_event['value']
    last_value = last_event['value']
    if (this_time-last_time).total_seconds() > 1:
        return this_event, 1
    else:
        if this_value > last_value:
            return this_event, 0
        else:
            return last_event, 0


def plot_xyz(raw_data, infos):

    raw_data = raw_data[:,:3]
    data_processed = processing_data_by_agreated(raw_data, 1)
    list_x = range(len(data_processed[:,0]))
    list_g = []
    data = data_processed - OFFSET
    y_level1, y_level2 = [], []
    for line in data:
        list_g.append(np.sqrt(sum(np.square(line))))
    max_g = max(list_g)
    mean_g = sum(list_g)/len(list_g)
    level1 = max_g*0.70
    level2 = mean_g
    for i in list_x:
        y_level1.append(level1)
        y_level2.append(level2)
    plt.plot(list_x, data[:,0], color = 'purple')
    plt.plot(list_x, data[:, 1], color='green')
    plt.plot(list_x, data[:, 2], color='yellow')
    plt.plot(list_x, list_g, color = 'red')
    plt.plot(list_x, y_level1, color = 'blue')
    plt.plot(list_x, y_level2, color = 'blue')
    plt.legend(('x', 'y', 'z','g'),
               loc='lower right')
    plt.title(infos)
    plt.show()

def main():

    level,  time = 'constant', '9h14'
    level , time = 'severe', '9h54'
    level , time = 'moyenne', '9h45'
    #level, time = 'faible', '9h28'
    raw_data, infos = get_raw_data(level, time)
    #crash_detection_from_raw_data(raw_data)
    #multi_plots(level, time)
    #movement_stop_detection(raw_data)
    #forward_backward_stop_detection(raw_data)
    #plots_of_movement(level, time)
    #all_detections(raw_data)
    processing_data_by_agreated_with_time(raw_data, 10)

def all_plot():

    level,  time = 'constant', '9h14'
    level , time = 'severe', '9h54'
    level , time = 'moyenne', '9h45'
    #level, time = 'faible', '9h28'
    raw_data, infos = get_raw_data(level, time)
    raw_data = simple_data_processing(raw_data[:, :])
    raw_data = raw_data[:, :3]
    #raw_data1 = processing_data_by_agreated(raw_data, 1)
    #raw_data2 = processing_data_by_agreated(raw_data, 4)
    #raw_data3 = processing_data_by_agreated(raw_data, 8)
    #raw_data4 = processing_data_by_agreated(raw_data, 16)
    raw_data1 = raw_data
    raw_data2 = processing_data_by_agreated_and_using_kalman_filter(raw_data, 1)
    raw_data3 = processing_data_by_agreated_and_using_kalman_filter(raw_data, 4)
    raw_data4 = processing_data_by_agreated_and_using_kalman_filter(raw_data, 8)
    array_raw_data = [[raw_data1, raw_data2], [raw_data3, raw_data4]]
    _, axarr = plt.subplots(2, 2)
    for i, raw_data_list in enumerate(array_raw_data):
        for j, raw_data in enumerate(raw_data_list):
            list_x = range(len(raw_data[:, 0]))
            list_g = []
            data = raw_data
            for line in data:
                list_g.append(np.sqrt(sum(np.square(line[:3]))))
            #axarr[i, j].plot(list_x, data[:, 0], color='purple')
            #axarr[i, j].plot(list_x, data[:, 1], color='green')
            axarr[i, j].plot(list_x, data[:, 2], color='yellow')
            #axarr[i, j].plot(list_x, list_g, color='red')
            #axarr[i, j].legend(('x', 'y', 'z', 'g'),
             #          loc='lower right')
    #plt.title(infos, loc = 'center')
    plt.show()

if __name__ == "__main__":
     #all_plot()
    #main()
    import pykalman













