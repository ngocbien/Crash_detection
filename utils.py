#!/usr/local/bin/python3
import config
import numpy as np
import math


def degree_to_radian(x):

    return x * math.pi / 180

def calcul_g_value(line):

    """

    :param array: line  is a vector of accelerometer of three axes in numpy array or
    4 axes with the last is datetime
    :return: g value
    """
    if sum(line[:3]) >= 0:
        sign_g = 1
    else:
        sign_g = -1
    return  np.sqrt(sum(np.square(line[:3]))) * sign_g

def accident_level(time_, value_g):
    """
    :param value_g:
    :return: message of sudden braking and accident level
    """
    if value_g < config.LEVEL_LIST[0]:
        return None
    dictionary = {'value': "%.2f" % value_g, 'datetime': time_}
    for index in range(len(config.LEVEL_LIST)-1):
        if value_g >= config.LEVEL_LIST[index] and value_g < config.LEVEL_LIST[index +1]:
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

def accident_level_with_small_scale(time_, value_g):
    """
     :param value_g:
     :return: dictionnary  about all infos of accident
     """
    if value_g < config.SMALL_SCALE_LEVEL_LIST[1]: #index 0 to detect sudden braking is not use in this case
        return None
    dictionary = {'value': "%.2f" % value_g, 'datetime': time_}
    for index in range(len(config.LEVEL_LIST) - 1):
        if value_g >= config.SMALL_SCALE_LEVEL_LIST[index] \
                and value_g < config.SMALL_SCALE_LEVEL_LIST[index + 1]:
            dictionary['level'] = index
            if index > 0 and index < 5:
                dictionary['message'] = 'MILD ACCIDENT'
            if index >= 5 and index < 9:
                dictionary['message'] = 'MEDIUM ACCIDENT'
            if index >= 9:
                dictionary['message'] = 'SEVERE ACCIDENT'
    return dictionary

def accident_and_sudden_braking_level_with_small_scale(time_, value_g):
    """
     :param value_g:
     :return: message of sudden braking and accident level
     """
    if value_g < config.SMALL_SCALE_LEVEL_LIST[0]:
        return None
    dictionary = {'value': "%.2f" % value_g, 'datetime': time_}
    for index in range(len(config.LEVEL_LIST) - 1):
        if value_g >= config.SMALL_SCALE_LEVEL_LIST[index] \
                and value_g < config.SMALL_SCALE_LEVEL_LIST[index + 1]:
            dictionary['level'] = index
            if index == 0:
                dictionary['message'] = 'SUDDEN BRAKING'
            if index > 0 and index < 5:
                dictionary['message'] = 'MILD ACCIDENT'
            if index >= 5 and index < 9:
                dictionary['message'] = 'MEDIUM ACCIDENT'
            if index >= 9:
                dictionary['message'] = 'SEVERE ACCIDENT'
    return dictionary

def accident_level_with_agrate(time_, value_g, M):
    """
    :param value_g:
    :return: message of sudden braking and accident level
    """
    dict_of_scale_value = {1: 1, 2: 1.2, 4: 2, 8: 3, 10: 4, 16: 5}
    if M in dict_of_scale_value:
        scale = dict_of_scale_value[M]
    else:
        scale = M/2
    scale = 1
    value_g = value_g * scale # make g bigger because agrate make down
    if value_g < config.LEVEL_LIST[0]:
        return None
    dictionary = {'value': "%.2f" % value_g, 'datetime': time_}
    for index in range(len(config.LEVEL_LIST)-1):
        if value_g >= config.LEVEL_LIST[index] and value_g < config.LEVEL_LIST[index +1]:
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

def filter_accident(last_accident, this_accident):
    """
    :param last_accident: is a dictionnary of infos about last accident:
    'datetime': datetime_of_accident, 'g_value': force in g, 'level': accident_level_of range 1 to 12
    :param this_accident: same format of last_accident
    :return: last_accident, None if the difference of time of these accident is less than 1 second and
     the level of this accident is smaller than last_accident. Else, return last_accident, and this accident.
    """
    if not last_accident:
        return None, this_accident
    difference_time = (this_accident['datetime'] - last_accident['datetime']).total_seconds()
    is_bigger_accident = this_accident['value'] > last_accident['value']
    if difference_time < 1:
        if  is_bigger_accident:
            return  None, this_accident
        else:
            return last_accident, None
    else:
        return last_accident, this_accident

def filter_movement_state(last_state, this_state):
    """
    :param last_accident: is a dictionnary of infos about last accident:
    'datetime': datetime_of_accident, 'g_value': force in g, 'level': accident_level_of range 1 to 12
    :param this_accident: same format of last_accident
    :return: last_accident, None if the difference of time of these accident is less than 1 second and
     the level of this accident is smaller than last_accident. Else, return last_accident, and this accident.
    """
    if not last_state:
        return None, this_state
    difference_time = (this_state['datetime'] - last_state['datetime']).total_seconds()
    is_bigger_accident = this_accident['value'] > last_accident['value']
    if difference_time < 1:
        if  is_bigger_accident:
            return  None, this_accident
        else:
            return last_accident, None
    else:
        return last_accident, this_accident

def find_movement_state(g_value, time_):
    """
    :param g_value: force in g
    :param time_: datetime of evenement
    :return: dictionnary of evemenent
    """
    abs_g = abs(g_value)
    dictionary = {'value': "%.2f" % g_value, 'datetime': time_}
    if abs_g < config.STOP_LEVEL:
        state = 0
    else:
        state = 1
    dictionary['state': state]
    return dictionary

def find_all_state(g_value, time_):
    """
    :param g_value: quantity of value of the force in g
    :return: dict_of_all_message
    """
    abs_g = abs(g_value)
    dictionary = {'value': "%.2f" % g_value, 'datetime': time_}
    time_print = time_.replace(microsecond=0)
    for index in range(len(LIST_TO_DETECT_ALL_STATE)-1):
        if abs_g >= LIST_TO_DETECT_ALL_STATE[index] and abs_g < LIST_TO_DETECT_ALL_STATE[index +1]:
            if index == 0:
                raw_index = 0
                dictionary['message'] = config.STOP_MSG
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
    :param list_of_state: list of state of value 0, 1, 2, 3, 4,
     which match with STOP BACKWARD FORWARD SUDDEN_BRAKING,
    and ACCIDENT
    :param raw_level: one of above level
    :return: new list
    """
    list_of_state.append(raw_level)
    if len(list_of_state) <= MAX_LENGTH:
        return list_of_state
    else:
        return list_of_state[1:]

