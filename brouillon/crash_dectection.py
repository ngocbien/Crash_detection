"""
Data is a csv file where we have time, id_name, ...
values of accelerometer is float( mesured by g unit)
gyros is float( I think it is by radian unit)
"""
import numpy as np
import pandas as pd

path = 'data/dataset.csv'
data_frame = pd.read_csv(path)

def accident_level(value_g):

    if value_g >=6 and value_g <20:
        return 'Mild Accident'
    elif value_g >= 20 and value_g <40:
        return 'Medium Accident'
    elif value_g >= 40:
        return 'Severe Accident'

def distance3D():
    pass

def print_some_line(data_frame):

    list_name = list(data_frame)[5:11]
    print('list name: ', list_name)
    all_vectors = pd.DataFrame(data_frame, columns = list_name)
    for index, line in all_vectors.iterrows():
        print(line)
        if index > 5:
            break


def simple_method_detection(data_frame):
    """
    :param data_frame: a pandas data frame
    :return: Message included time of accidence, if exist in the round of 10 second
    else: return None
    """
    list_name = ['acceleration_x', 'acceleration_y', 'acceleration_z']
    message = []
    all_vector = pd.DataFrame(data_frame, columns=list_name)
    for index, line in all_vector.iterrows():
        if index >1:
            value_g = np.sqrt(sum(np.square(line - last_line)))
            if value_g >= 6:
                level = accident_level(value_g)
                message_ = 'ACCIDENT: level {} at {} {}'.format(level,
                    data_frame.loc[index, 'date'],
                      data_frame.loc[index, 'time'])
                message.append(message_)
                print(message_)
        last_line = line
    if len(message) == 0:
        return None
    return message



#sample_method_detection(data_frame)
#print_some_line(data_frame)