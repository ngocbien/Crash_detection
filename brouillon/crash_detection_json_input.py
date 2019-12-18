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
PLOT = True

if PLOT:
    import matplotlib.pyplot as plt


SENSITIVITY_ACC = 16384 # for accelerometer of 16bit: 2^16, we suppose that our captor is use for
#range +/- 2g
SENSITIVITY_GYRO = 131 #see https://www.invensense.com/products/motion-tracking/6-axis/mpu-6050/
LEVEL0_LOW  = 0.8 #premier niveau de brusque freinage
LEVEL1_LOW  = 4 # normally is 4 but we modified to get some accident nofitication
LEVEL1_HIG  = 20
LEVEL2_HIG  = 40
LEVEL3_LOW  = 50
LEVEL1_LOW  = 2.1 # normally is 4 but we modified to get some accident nofitication

DATA_PATH = 'data/document.json'
DATA_PATH = 'data/accelerometer_gyroscope_9_crashes.json'
DATA_PATH = 'data/dataset_choc_faible_1.txt'
#DATA_PATH = 'data/constant_dataset.txt'
#PREQUENCY = 50 #This is the frequecy of return data by sec. MUST BE CHANGED TO ADAPT OUR DEVICE
PREQUENCY = 3
ALPHA = 1/10 # this is a tuning parameter to combine Acc and Gyro for a more precise value of Acc
#this value 1/10 is not good because it is too sensitive to the Gyro
ALPHA = 7/10 #not good
#ALPHA = 19/20 #good value
#ALPHA = 1

def get_data():

    data_processed = []
    with open(DATA_PATH) as json_data:
        d = json.load(json_data)
    for line in d:
        data = line['smn'][0]['data'][0]
        #print(data)
        #break
        acc = data['accelerometer'][0]
        acc_x = float(acc['x'])/SENSITIVITY_ACC
        acc_y = float(acc['y'])/SENSITIVITY_ACC
        acc_z = float(acc['z'])/SENSITIVITY_ACC
       # acc_z = float(data['accelerometer']['z'])
        data_processed.append([acc_x, acc_y, acc_z])
    return data_processed

def test_data(data_processed):

    max_ = 0
    min_ = 0
    seuil = 0.3
    for line in data_processed:
        if max(line)> max_:
            max_ = max(line)
        if min_ >min(line):
            min_ = min(line)
    print('max and min are ' , max_, min_)
    num_inf = 0
    num_sup = 0
    max_g = 0
    min_g = 10 # To be sure that there exists some values that smaller than this one
    print('length of data set is {}'.format(len(data_processed)))
    for index, line in enumerate(data_processed):
        if index >0:
             value_g = np.sqrt(sum(np.square(line - last_line)))
             if value_g > max_g:
                 max_g = value_g
             if value_g < min_g:
                 min_g = value_g
             last_line = line
             if value_g >seuil:
                 num_sup += 1
             else:
                 num_inf += 1
        else:
            last_line = line
    print('with level {}, there are {} down and {} up'.format(
        seuil, num_inf, num_sup
    ))
    print('min and max of g value are {} and {} respectively'.format(min_g, max_g))

def accident_level(value_g):
    """
    :param value_g:
    :return: message of sudden braking and accident level
    """
    if value_g >= LEVEL0_LOW and value_g < LEVEL1_LOW:
        return 'Sudden Braking'
    if value_g >= LEVEL1_LOW and value_g < LEVEL1_HIG:
        return 'Mild Accident'
    elif value_g >= LEVEL1_HIG and value_g < LEVEL2_HIG:
        return 'Medium Accident'
    elif value_g >= LEVEL2_HIG and value_g < LEVEL3_LOW:
        return 'Severe Accident'
    elif value_g >= LEVEL3_LOW:
        return 'Very Severe Accident'

def simple_method_detection():
    """
    :param data_frame: a pandas data frame
    :return: Message included time of accidence, if exist in the round of 10 second
    else: return None
    """
    all_vector = get_data()
    all_vector = np.asarray(all_vector, dtype=np.float32)
    message = []
    for index, line in enumerate(all_vector):
        if index > LEVEL1_LOW:
            value_g = np.sqrt(sum(np.square(line - last_line)))
            if value_g >= LEVEL1_LOW:
                level = accident_level(value_g)
                message_ = 'ACCIDENT: level {}'.format(level)
                message.append(message_)
                print(message_)
        last_line = line
    if not x:
        return None
    print('{} accidents detected'.format(len(message)))
    return message

def get_accelerometer_and_gyroscope_data(json_data):
    """
    :return: Accelometer  and gyroscope of this data in numpy array
    """
    index_count = 0
    for line in json_data:
        data = line['smn'][0]['data'][0]
        acc = data['accelerometer'][0]
        acc_x = float(acc['x'])/SENSITIVITY_ACC
        acc_y = float(acc['y'])/SENSITIVITY_ACC
        acc_z = float(acc['z'])/SENSITIVITY_ACC
        gyro = data['gyroscope'][0]
        gyro_x = float(gyro['x'])/SENSITIVITY_GYRO
        gyro_y = float(gyro['y'])/SENSITIVITY_GYRO
        gyro_z = float(gyro['z'])/SENSITIVITY_GYRO
        new_row = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        if index_count == 0:
            data_processed = np.asanyarray(new_row)
        else:
            data_processed = np.vstack([data_processed, new_row])
        index_count += 1
    return data_processed

def reduce_noise_in_data(raw_data):
    """
    :param alpha: Tuning parameter
    :return: numpy array of  x,y,z estimated
    all  calculations are from http://www.starlino.com/imu_guide.html
    """
    first_row = raw_data[0,:3]
    estimated_acc = np.asanyarray(first_row)
    last_estimated_acc = estimated_acc
    for i in range(1, len(raw_data[:,0])):
        last_Axz = math.atan2(last_estimated_acc[0], last_estimated_acc[2]) #Ax, Az
        last_Ayz = math.atan2(last_estimated_acc[1], last_estimated_acc[2]) #Ay, Az
        rate_Axz = raw_data[i, 5]-raw_data[i-1, 5]
        rate_Ayz = raw_data[i, 4]-raw_data[i-1, 4]
        this_Axz = last_Axz + rate_Axz * PREQUENCY
        this_Ayz = last_Ayz + rate_Ayz * PREQUENCY
        RxGro = math.sin(this_Axz)/math.sqrt(1 + math.pow(math.cos(this_Axz)*math.tan(this_Ayz), 2))
        RyGro = math.sin(this_Ayz)/math.sqrt(1 + math.pow(math.cos(this_Ayz)*math.tan(this_Axz), 2))
        if last_estimated_acc[2] >=0:
            RzGro = math.sqrt(1- RxGro * RxGro - RyGro * RyGro)
        else:
            RzGro = (-1)*math.sqrt(1 - RxGro * RxGro - RyGro * RyGro)
        RGro = np.asarray([RxGro, RyGro, RzGro])
        new_row = np.asarray(ALPHA*raw_data[i,:3]+(1-ALPHA)*RGro)
        #print(new_row)
        estimated_acc = np.vstack([estimated_acc, new_row])
        last_estimated_acc = estimated_acc[i,:]
    return estimated_acc

def crash_detection_with_reduced_noise_data(estimated_data_acc):
    """
    In one second, there is max an accident
    :return: All messages about accident
    """
    message = []
    last_evenement_index = -PREQUENCY-1 #for sure to save the first accident
    for index, line in enumerate(estimated_data_acc):
        if index > 0:
            value_g = np.sqrt(sum(np.square(line - last_line)))
            if value_g >= LEVEL0_LOW and value_g < LEVEL1_LOW and (index-last_evenement_index)> PREQUENCY:
                print('SUDDEN BRAKING')
                last_evenement_index = index
            if value_g >= LEVEL1_LOW and (index-last_evenement_index) > PREQUENCY:
                level = accident_level(value_g)
                #message_ = 'ACCIDENT: level {} and {:.2f}'.format(level, value_g)
                message_ = 'ACCIDENT: level {}'.format(level)
                message.append(message_)
                print(message_)
                last_evenement_index = index
        last_line = line
    if not message:
        return None
    print('{} accidents detected'.format(len(message)))
    return message

def get_message(last_message, last_status,  this_status):
    """
    :param last_message: The message about the status of vehicle the last time
    :param last_status, this_status: 0 if stop, 1 if mouvement
    :return : new message
    """
    if '.'*20  in last_message:
        last_message = last_message.strip('.')
        #print(last_message)
    if last_status == 1:
        if this_status == 1:
            return last_message +'.'
        else:
            return 'Vehicle in stop'
    else:
        if this_status == 1:
            return 'vehicle in movement'
        else:
            return last_message + '.'

def mouvement_detection(estimated_data_acc):

    LEVEL_STOP = 0.15
    TIME_PRINT = 1
    last_index_print = -TIME_PRINT - 1
    last_status, this_status = 0, 0 # the first state is consider that
    last_message = 'vehicle in stop'
    number_strip = 1 # nombre de trajets
    count_time_stop = 0
    boolean = False
    for index, line in enumerate(estimated_data_acc):
        if index > 0:
            value_g = np.sqrt(sum(np.square(line - last_line)))
            if value_g >= LEVEL_STOP and (index-last_index_print) > TIME_PRINT:
                this_status = 1
                message = get_message(last_message, last_status, this_status)
                last_message = message
                print(message, end = '\r')
                time.sleep(.5)
                last_index_print = index
            if value_g < LEVEL_STOP and (index-last_index_print) > TIME_PRINT:
                this_status = 0
                if last_status == 1:
                    boolean = 1 # pour commencer à calculer le temps d'arrêt
                    count_time_stop = 0
                if boolean:
                    count_time_stop += 1
                if count_time_stop > PREQUENCY:# si le vehicule s'arrête plus d'une second, ça va
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

def plot_all_g(data_processed):

    #import matplotlib.pyplot as plt
    print('length of data set is {}'.format(len(data_processed)))
    list_y = []
    for index, line in enumerate(data_processed):
        if index > 0:
            value_g = np.sqrt(sum(np.square(line - last_line)))
            list_y.append(value_g)
        last_line = line
    list_x = range(len(list_y))
    print('max = {} and min = {}'.format(max(list_y), min(list_y)))
    plt.plot(list_x, list_y)
    plt.show()

def plot_g_value_list(list):

    list_x = range(len(list))
    plt.plot(list_x, list)
    plt.show()


def some_test_code():

    pass

def get_list_of_g(estimated_data_acc):
    """
    ATTENTION:
    :param data: is 3D accelerometer clean data, with a given frequen
    :return: numpy array data of acceleration or decceleration
    """
    list_ = []
    for index, line in enumerate(estimated_data_acc):
        if index > 0:
            vector_g = sum(line-last_line)
            if vector_g >= 0:
                sign_g = 1
            else:
                sign_g = -1
            value_g = np.sqrt(sum(np.square(line - last_line)))*sign_g
            list_.append(value_g)

        last_line = line
    return list_



def main():

    data = open(DATA_PATH)
    print('Opening data file at {}'.format(DATA_PATH))
    json_data = json.load(data)
    raw_data = get_accelerometer_and_gyroscope_data(json_data)
    estimated_data_acc = reduce_noise_in_data(raw_data)
    crash_detection_with_reduced_noise_data(estimated_data_acc)
    mouvement_detection(estimated_data_acc)
    test_data(estimated_data_acc)
   # plot_all_g(estimated_data_acc)
    plot_g_value_list(get_list_of_g(estimated_data_acc))



if __name__ == "__main__":
    main()




