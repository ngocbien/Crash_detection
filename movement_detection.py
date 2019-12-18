#!/usr/local/bin/python3
import utils
import config
import data_processing
def movement_detection(raw_data, list_of_state = None):
    """
    :param raw_data: is a dictionnary {'datetime': datetime, 'accelerometer': [{'x': acc_x, 'y': acc_y,
    'z': acc_z}], 'gyroscope': [{'x': gyro_x, 'y': gyro_y, 'z': gyro_z}]}
    last_state = is a dictionary {'datetime': time_of_evenement,  'g_value': force in g, 'state': level_of_evement}
    state : 0: STOP, 1: FORWARD, -1: BACKWARD
    :return: list a dictionnarys of states [{}, {}, {},...]
    """
    raw_data = data_processing.get_raw_data(raw_data)
    data_processed = data_processing.processing_by_kalman_filter(raw_data)
    num_event = 0
    list_of_states = []
    for  index, line in enumerate(data_processed):
        print('function is in construction ')
        break
        g_value = utils.calcul_g_value(line[:3])
        this_time = line[-1]
        this_state = utils.find_movement_state(g_value, this_time)
        if not list_of_state:
            list_of_states.append(this_state)
        else:
            if is_tuning_state(state, list_of_state):
                if tuning_step > SKIP_STEP and save_state == find_state(g_value):
                    state = find_state(g_value)
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
                        time_start_event = time_

