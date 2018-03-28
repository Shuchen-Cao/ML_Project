# this version is used to estimate the following gait parameters:
# GRF, COP, KAM, FPA, ankle flexion angle, knee flexion angle, hip flexion angle
# pelvis angles x,y,z, lateral trunk sway angle, heel strike events, toe off events


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from ParamProcessorClass import ParamProcessor
from numpy.linalg import norm
from GaitParameterClass import GaitParameter

file_date = '20171228\\'
data_path = 'D:\Tian\Research\Projects\ML Project\processed_data\\' + file_date

gait_names = ['normal', 'toeout', 'toein', 'largeSW', 'largeTS', 'toein_largeSW', 'toeout_largeSW']
gait_num = gait_names.__len__()

with open(data_path + 'subject_info.csv') as subject_info_file:
    reader = csv.reader(subject_info_file)
    for row in reader:  # there is only one row in the csv file
        subject_info = row

# dynamic_var = locals()  # dynamically name the Object and csv file

for gait_name in gait_names:
    # gait_data_len = gait_data.shape[0]
    gait_file_path = data_path + 'GaitData\gait_' + gait_name + '.csv'
    parameter_processor = ParamProcessor(gait_file_path, float(subject_info[1]), float(subject_info[2]))

    force = parameter_processor.get_force()
    COP = parameter_processor.get_COP()
    KAM = parameter_processor.get_KAM()
    strike = parameter_processor.get_heel_strike_event()
    off = parameter_processor.get_toe_off_event()
    step_width = parameter_processor.get_step_width()
    FPA = parameter_processor.get_FPA()
    trunk_swag = parameter_processor.get_trunk_swag()
    pelvis_angle = parameter_processor.get_pelvis_angle()
    ankle_flexion_angle = parameter_processor.get_ankle_flexion_angle()
    knee_flexion_angle = parameter_processor.get_knee_flexion_angle()
    hip_flexion_angle = parameter_processor.get_hip_flexion_angle()

    # # check data
    # parameter_processor.check_KAM(KAM)
    # parameter_processor.check_strikes_off(strike, off)
    # parameter_processor.check_step_width(step_width)
    # parameter_processor.check_FPA(FPA)
    # parameter_processor.check_trunk_swag(trunk_swag)
    # parameter_processor.check_pelvis_angle(pelvis_angle)
    # parameter_processor.check_ankle_flexion_angle(ankle_flexion_angle)
    # parameter_processor.check_knee_flexion_angle(knee_flexion_angle)
    # parameter_processor.check_hip_flexion_angle(hip_flexion_angle)

    # plt.show()

    # transfer matrix to DataFrame
    raw_data = np.column_stack([force, COP, KAM, strike, off, step_width, FPA, pelvis_angle, trunk_swag,
                                ankle_flexion_angle, knee_flexion_angle, hip_flexion_angle])
    data = pd.DataFrame(raw_data, columns=[
        'f_1_x', 'f_1_y', 'f_1_z', 'f_2_x', 'f_2_y', 'f_2_z', 'c_1_x', 'c_1_y', 'c_1_z', 'c_2_x', 'c_2_y', 'c_2_z',
        'KAM_l', 'KAM_R', 'strike_l', 'strike_r', 'off_l', 'off_r',
        'step_width', 'FPA_l', 'FPA_r', 'pelvis_angle_x', 'pelvis_angle_y', 'pelvis_angle_z', 'trunk_swag',
        'ankle_flexion_angle_l', 'ankle_flexion_angle_r', 'knee_flexion_angle_l', 'knee_flexion_angle_r',
        'hip_flexion_angle_l', 'hip_flexion_angle_r'])

    object_name = 'params_of_' + gait_name  # the name of the GaitData object and file
    current_class = GaitParameter(object_name)  # initial the object
    save_path = data_path + 'GaitParameters\\'
    current_class.set_data(data)  # pass the data to the matrix
    # save as csv
    current_class.save_as_csv(save_path)  # save as csv
    current_class.clear_old_csv(save_path, object_name)  # delete the former file
