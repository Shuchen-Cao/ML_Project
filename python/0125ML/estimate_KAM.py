# mainly use the right leg data because it contains less noise

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter

import tensorflow as tf

from KAMDataClass import KAMData
import csv


file_date = '20171229\\'
processed_data_path = 'D:\Tian\Research\Projects\ML Project\processed_data\\' + file_date

# validation_set_len = 10000  # the length of validation
test_set_len = 500  # the length of data for test
gait_num = 7
cut_zero_data = False  # cut data of KAM == 0

# gait names in the sequence of the experiment
if file_date == '20171229\\':
    gait_names = ('normal', 'toeout', 'toein', 'largeSW', 'largeTS',
                  'toein_largeSW', 'toeout_largeSW', 'change_location')
else:
    gait_names = ('normal', 'toeout', 'toein', 'largeSW', 'largeTS', 'toein_largeSW', 'toeout_largeSW')

with open(processed_data_path + 'subject_info.csv') as subject_info_file:
    reader = csv.reader(subject_info_file)
    for row in reader:  # there is only one row in the csv file
        subject_info = row
dynamic_var = locals()  # dynamically name the Object and csv file
subject_name = 'subject_' + subject_info[0]
dynamic_var[subject_name] = KAMData(float(subject_info[1]), float(subject_info[2]), subject_info[0])

# initialize y
l_knee_l_training = np.zeros([0, 3])
l_knee_r_training = np.zeros([0, 3])
r_knee_l_training = np.zeros([0, 3])
r_knee_r_training = np.zeros([0, 3])
l_cop_training = np.zeros([0, 3])
r_cop_training = np.zeros([0, 3])
l_force_training = np.zeros([0, 3])
r_force_training = np.zeros([0, 3])

l_knee_l_testing = np.zeros([0, 3])
l_knee_r_testing = np.zeros([0, 3])
r_knee_l_testing = np.zeros([0, 3])
r_knee_r_testing = np.zeros([0, 3])
l_cop_testing = np.zeros([0, 3])
r_cop_testing = np.zeros([0, 3])
l_force_testing = np.zeros([0, 3])
r_force_testing = np.zeros([0, 3])

# initialize x
x_column_num = 24
acc_training = np.zeros([0, x_column_num])
acc_testing = np.zeros([0, x_column_num])
acc_column_names = ['trunk_acc_x', 'trunk_acc_y', 'trunk_acc_z', 'pelvis_acc_x', 'pelvis_acc_y', 'pelvis_acc_z',
                    'l_thigh_acc_x', 'l_thigh_acc_y', 'l_thigh_acc_z',
                    'r_thigh_acc_x', 'r_thigh_acc_y', 'r_thigh_acc_z',
                    'l_shank_acc_x', 'l_shank_acc_y', 'l_shank_acc_z',
                    'r_shank_acc_x', 'r_shank_acc_y', 'r_shank_acc_z',
                    'l_foot_acc_x', 'l_foot_acc_y', 'l_foot_acc_z', 'r_foot_acc_x', 'r_foot_acc_y', 'r_foot_acc_z']

# set filter parameters
filter_order = 4
wn_force = 50 / (1000 / 2)  # unified frequency
b_force_plate, a_force_plate = butter(filter_order, wn_force, btype='low')
wn_marker = 6 / 50
b_marker, a_marker = butter(filter_order, wn_marker, btype='low')

for gait_name in gait_names:
    gait_file_path = processed_data_path + 'GaitData\gait_' + gait_name + '.csv'
    gait_data = pd.read_csv(gait_file_path)
    gait_data_len = gait_data.shape[0]

    if cut_zero_data:
        gait_data = gait_data[gait_data.f_2_z != -1.0]

    # initialize training y
    l_knee_l_training = np.concatenate((l_knee_l_training, gait_data.as_matrix(
        columns=['l_knee_l_x', 'l_knee_l_y', 'l_knee_l_z'])[0: -test_set_len]))
    l_knee_r_training = np.concatenate((l_knee_r_training, gait_data.as_matrix(
        columns=['l_knee_r_x', 'l_knee_r_y', 'l_knee_r_z'])[0: -test_set_len]))
    r_knee_l_training = np.concatenate((r_knee_l_training, gait_data.as_matrix(
        columns=['r_knee_l_x', 'r_knee_l_y', 'r_knee_l_z'])[0: -test_set_len]))
    r_knee_r_training = np.concatenate((r_knee_r_training, gait_data.as_matrix(
        columns=['r_knee_r_x', 'r_knee_r_y', 'r_knee_r_z'])[0: -test_set_len]))
    l_cop_training = np.concatenate((l_cop_training, gait_data.as_matrix(
        columns=['c_1_x', 'c_1_y', 'c_1_z'])[0: -test_set_len]))
    r_cop_training = np.concatenate((r_cop_training, gait_data.as_matrix(
        columns=['c_2_x', 'c_2_y', 'c_2_z'])[0: -test_set_len]))
    l_force_training = np.concatenate((l_force_training, gait_data.as_matrix(
        columns=['f_1_x', 'f_1_y', 'f_1_z'])[0: -test_set_len]))
    r_force_training = np.concatenate((r_force_training, gait_data.as_matrix(
        columns=['f_2_x', 'f_2_y', 'f_2_z'])[0: -test_set_len]))

    # initialize testing y
    l_knee_l_testing = np.concatenate((l_knee_l_testing, gait_data.as_matrix(
        columns=['l_knee_l_x', 'l_knee_l_y', 'l_knee_l_z'])[-test_set_len:, :]))
    l_knee_r_testing = np.concatenate((l_knee_r_testing, gait_data.as_matrix(
        columns=['l_knee_r_x', 'l_knee_r_y', 'l_knee_r_z'])[-test_set_len:]))
    r_knee_l_testing = np.concatenate((r_knee_l_testing, gait_data.as_matrix(
        columns=['r_knee_l_x', 'r_knee_l_y', 'r_knee_l_z'])[-test_set_len:]))
    r_knee_r_testing = np.concatenate((r_knee_r_testing, gait_data.as_matrix(
        columns=['r_knee_r_x', 'r_knee_r_y', 'r_knee_r_z'])[-test_set_len:]))
    l_cop_testing = np.concatenate((l_cop_testing, gait_data.as_matrix(
        columns=['c_1_x', 'c_1_y', 'c_1_z'])[-test_set_len:]))
    r_cop_testing = np.concatenate((r_cop_testing, gait_data.as_matrix(
        columns=['c_2_x', 'c_2_y', 'c_2_z'])[-test_set_len:]))
    l_force_testing = np.concatenate((l_force_testing, gait_data.as_matrix(
        columns=['f_1_x', 'f_1_y', 'f_1_z'])[-test_set_len:]))
    r_force_testing = np.concatenate((r_force_testing, gait_data.as_matrix(
        columns=['f_2_x', 'f_2_y', 'f_2_z'])[-test_set_len:]))

    # initialize training x
    acc_training = np.concatenate((acc_training, gait_data.as_matrix(columns=acc_column_names)[0: -test_set_len]))

    # initialize testing x
    acc_testing = np.concatenate((acc_testing, gait_data.as_matrix(columns=acc_column_names)[-test_set_len:]))

# as for knee_side, 0 represents left knee, 1 represents right knee
# as for data_set, 0 for training, 1 for testing
dynamic_var[subject_name].set_data(l_knee_l_training, l_knee_r_training,
                                   l_force_training, l_cop_training, data_set=0, knee_side=0)
dynamic_var[subject_name].set_data(r_knee_l_training, r_knee_r_training,
                                   r_force_training, r_cop_training, data_set=0, knee_side=1)
dynamic_var[subject_name].set_data(l_knee_l_testing, l_knee_r_testing,
                                   l_force_testing, l_cop_testing, data_set=1, knee_side=0)
dynamic_var[subject_name].set_data(r_knee_l_testing, r_knee_r_testing,
                                   r_force_testing, r_cop_testing, data_set=1, knee_side=1)

KAM_data_l_training = dynamic_var[subject_name].get_KAM(data_set=0, knee_side=0)
KAM_data_r_training = dynamic_var[subject_name].get_KAM(data_set=0, knee_side=1)
KAM_data_l_testing = dynamic_var[subject_name].get_KAM(data_set=1, knee_side=0)
KAM_data_r_testing = dynamic_var[subject_name].get_KAM(data_set=1, knee_side=1)


# plt.figure()
# plt.plot(KAM_data_l_training)  # check the KAM data
# plt.figure()
# plt.plot(KAM_data_l_testing)  # check the KAM data
# plt.show()


def method_evaluation_right(model):
    model.fit(acc_training, KAM_data_r_training)
    score = model.score(acc_testing, KAM_data_r_testing)
    result = model.predict(acc_testing)
    plt.figure()
    plt.plot(KAM_data_r_testing, 'b', label='true value')
    plt.plot(result, 'r', label='predicted value')
    for i_gait in range(1, gait_num + 1):
        plt.plot((test_set_len * i_gait, test_set_len * i_gait), (-0.5, 0.5), 'y--')
    plt.title('score: %f' % score)
    plt.legend()


def method_evaluation_left(model):
    model.fit(acc_training, KAM_data_l_training)
    score = model.score(acc_testing, KAM_data_l_testing)
    result = model.predict(acc_testing)
    plt.figure()
    plt.plot(KAM_data_l_testing, 'b', label='true value')
    plt.plot(result, 'r', label='predicted value')
    for i_gait in range(1, gait_num + 1):
        plt.plot((test_set_len * i_gait, test_set_len * i_gait), (-0.5, 0.5), 'y--')
    plt.title('score: %f' % score)
    plt.legend()


# from sklearn import tree
# model_decision_tree = tree.DecisionTreeRegressor()
# method_evaluation(model_decision_tree)

# from sklearn import svm
# model_SVR = svm.SVR()
# method_evaluation(model_SVR)

# from sklearn import neighbors
# model_KNN = neighbors.KNeighborsRegressor()
# method_evaluation_left(model_KNN)
# method_evaluation_right(model_KNN)

from sklearn import ensemble
model_random_forest = ensemble.RandomForestRegressor()
method_evaluation_left(model_random_forest)
method_evaluation_right(model_random_forest)

# # check the acc data
# plt.figure()
# plt.plot(acc_training[:,  23])

plt.show()
