import json
import numpy as np

SENSITIVITY = 16384
LEVEL1_LOW = 0.95  # normally is 4 but we modified to get some accident nofitication


def format_data(data):

    data_processed = []
    for line in data:
        data = line['smn'][0]['data'][0]
        # print(type(data))
        acc = data['accelerometer'][0]
        acc_x = float(acc['x']) / SENSITIVITY
        acc_y = float(acc['y']) / SENSITIVITY
        acc_z = float(acc['z']) / SENSITIVITY
        # acc_z = float(data['accelerometer']['z'])
        data_processed.append([acc_x, acc_y, acc_z])
    return data_processed

def accident_level(value_g):
    return  value_g // LEVEL1_LOW

def simple_method_detection(acc_data):
    """
    ;:param json_payload:
    :return int
    """
    all_vector = format_data(acc_data)
    all_vector = np.asarray(all_vector, dtype=np.float32)
    crashes = []

    for index, line in  enumerate(all_vector):
        if index > 0:
            value_g = np.sqrt(sum(np.square(line - last_line)))
            if value_g >= LEVEL1_LOW:
                level = accident_level(value_g)
                crashes.append(level)
        last_line = line
    if not len(crashes):
        return 0
    return max(crashes)


if __name__ == '__main__':
    test_data_path = 'data/accelerometer_gyroscope_9_crashes.json'
    with open(test_data_path) as json_data:
        acc_data = json.load(json_data)

    print(simple_method_detection(acc_data))


