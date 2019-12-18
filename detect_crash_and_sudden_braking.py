#!/usr/local/bin/python3
import utils
import data_processing
import numpy as np
def crash_and_sudden_braking_detection(raw_data, last_accident=None):
    """

    :param raw_data: is a dictionnary {'datetime': datetime, 'accelerometer': [{'x': acc_x, 'y': acc_y,
    'z': acc_z}], 'gyroscope': [{'x': gyro_x, 'y': gyro_y, 'z': gyro_z}]}
    last_accident = is a dictionary {'datetime': time_of_evenement,  'g_value': force in g, 'level': level_of_accident}
    :return: list a dictionnarys of all accidents, example, [{}, {}, ...]
    dictionnary = {'value': value of force, 'level': level of accident/12, 'datetime': time when accident occcurs,
    'message': message about level of accident: sudden_braking -mild- medium- severe}
    Note that we can print all accident in list_of_accident, except the last one, which will be use to to compare
    with others accidents lately
    """
    list_of_accident = []
    data_processed = data_processing.processing_data_by_agreated(raw_data, M=5)
    for  line in data_processed:
        value_g = np.sqrt(sum(np.square(line[:3])))
        time_ = line[-1]
        this_accident = utils.accident_and_sudden_braking_level_with_small_scale(time_, value_g)
        if this_accident:
            last_accident, this_accident = utils.filter_accident(last_accident, this_accident)
            if last_accident and this_accident:
                list_of_accident.append(last_accident)
                last_accident = this_accident
            else:
                if not last_accident:
                    last_accident = this_accident
    return list_of_accident