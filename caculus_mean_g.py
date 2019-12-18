#!/usr/local/bin/python3
import utils
import data_processing
def mean_of_g(raw_data):
    """
    :param raw_data: is a dictionnary {'datetime': datetime, 'accelerometer': [{'x': acc_x, 'y': acc_y,
    'z': acc_z}], 'gyroscope': [{'x': gyro_x, 'y': gyro_y, 'z': gyro_z}]}
    :return: mean of force in g
    """
    total_g = 0
    data_processed = data_processing.reduce_bias(raw_data)
    for  index, line in enumerate(data_processed):
        value_g = utils.calcul_g_value(line[:3])
        total_g += value_g
    return total_g/(index + 1)

