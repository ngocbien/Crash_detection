Run crash_detection.py to detect accident.

Run detect_crash_and_sudden_braking.py to detect all crash and sudden braking

Run caculus_mean_g.py to get mean of the force from raw data.

raw_data is a data where each line is a dict of type: 

{'datetime': datetime, 'accelerometer': [{'x': acc_x, 'y': acc_y,
    'z': acc_z}], 'gyroscope': [{'x': gyro_x, 'y': gyro_y, 'z': gyro_z}]}
