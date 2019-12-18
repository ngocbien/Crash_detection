#!/usr/local/bin/python3
import config
import os
import json
import crash_detection
import caculus_mean_g
import data_processing
import pylab as plt
import numpy as np
import utils

def main():
    level, time = 'moyenne', '9h45'
    #level, time = 'faible', '9h28'
    file = config.DICT_DATASET[level][time]
    path = os.path.join('data', file)
    data = open(path)
    raw_data = json.load(data)
    raw_data = data_processing.get_raw_data(raw_data)
    raw_data1 = raw_data.copy()
    processed_data = data_processing.processing_by_kalman_filter(raw_data)
    print('processed data ', processed_data[0, :])
    print('mean in z acc processed ', mean(processed_data[:, 2]))
    print()
    print('raw data ', raw_data1[0, :])
    print('mean in z raw data', mean(raw_data1[:, 2]))
    return   raw_data1, processed_data

def mean(x):
    return sum(x)/len(x)
def plot_data():

    raw_data, processed_data = main()
    print(mean(raw_data[:,2]))
    print(mean(processed_data[:, 2]))
    print(raw_data[10, :])
    print(processed_data[10, :])
    _, axarr = plt.subplots(1, 3)
    x_raw = axarr[0].plot(raw_data[:, 0], color='g')
    x_processed = axarr[0].plot(processed_data[:, 0], color='r')
    y_raw = axarr[1].plot(raw_data[:, 1], color='g')
    y_processed = axarr[1].plot(processed_data[:, 1], color='r')
    z_raw = axarr[2].plot(raw_data[:, 2], color='g')
    z_processed = axarr[2].plot(processed_data[:, 2], color='r')
    plt.legend((x_raw, x_processed), ('raw', 'processed'),
              loc='lower right')
    plt.show()

def calcul_mean():
    level, time = 'moyenne', '9h45'
    level, time = 'faible', '9h28'
    level, time = 'constant', '9h10'
    file = config.DICT_DATASET[level][time]
    path = os.path.join('data', file)
    data = open(path)
    raw_data = json.load(data)
    print(caculus_mean_g.mean_of_g(raw_data))

def crash_detection_using_kalman_filter(last_accident = None):

    """
        :param raw_data: is a dictionnary {'datetime': datetime, 'accelerometer': [{'x': acc_x, 'y': acc_y,
        'z': acc_z}], 'gyroscope': [{'x': gyro_x, 'y': gyro_y, 'z': gyro_z}]}
        last_accident = is a dictionary {'datetime': time_of_evenement,  'g_value': force in g, 'level': level_of_accident}
        :return: list a dictionnarys of all accidents, example, [{}, {}, ...]
        dictionnary = {'value': value of force, 'level': level of accident/12, 'datetime': time when accident occcurs,
        'message': message about level of accident: mild- medium- severe}
        Note that we can print all accident in list_of_accident, except the last one, which will be use to to compare
        with others accidents lately
        """
    _, data_processed = main()
    list_of_accident = []
    for line in data_processed:
        value_g = np.sqrt(sum(np.square(line[:3])))
        time_ = line[-1]
        this_accident = utils.accident_level_with_small_scale(time_, value_g)
        if this_accident:
            last_accident, this_accident = utils.filter_accident(last_accident, this_accident)
            if last_accident and this_accident:
                list_of_accident.append(last_accident)
                last_accident = this_accident
            else:
                if not last_accident:
                    last_accident = this_accident
    return list_of_accident


def test_numpy():
    import numpy as np
    a = np.zeros((2,3), dtype=np.int)
    b = np.ones((2,3))
    a[:, :2] =  b[:, :2]
    print(a)
#plot_data()
#main()
list = crash_detection_using_kalman_filter()
for accident in list:
    #print(accident['message'], accident['level'], accident['value'])
    print(accident)




