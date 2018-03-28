# this version is used for testing the influence of different training sets data.
# for example, different subjects' data were used, different combination of gaits were used


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from KAMDataClass2 import KAMData
from sklearn import preprocessing

training_file_date = '20171229\\'
training_data_path = 'D:\Tian\Research\Projects\ML Project\processed_data\\' + training_file_date
testing_file_date = '20171229\\'
testing_data_path = 'D:\Tian\Research\Projects\ML Project\processed_data\\' + testing_file_date

test_set_len = 500  # the length of data for test
gait_num = 7
cut_zero_data = False  # cut data of KAM == 0

training_gait_names = ['normal', 'toeout', 'toein', 'largeSW',
                       'toein_largeSW', 'toeout_largeSW']
testing_gait_names = ['largeTS']

with open(testing_data_path + 'subject_info.csv') as subject_info_file:
    reader = csv.reader(subject_info_file)
    for row in reader:  # there is only one row in the csv file
        subject_info = row
dynamic_var = locals()  # dynamically name the Object and csv file
subject_name = 'subject_' + subject_info[0]

# initialize four KAMs
dynamic_var[subject_name + '_training' + '_left'] = \
    KAMData(float(subject_info[1]), float(subject_info[2]), subject_info[0], 'training', 'left')
dynamic_var[subject_name + '_training' + '_right'] = \
    KAMData(float(subject_info[1]), float(subject_info[2]), subject_info[0], 'training', 'right')
dynamic_var[subject_name + '_testing' + '_left'] = \
    KAMData(float(subject_info[1]), float(subject_info[2]), subject_info[0], 'testing', 'left')
dynamic_var[subject_name + '_testing' + '_right'] = \
    KAMData(float(subject_info[1]), float(subject_info[2]), subject_info[0], 'testing', 'right')


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
                    'l_foot_acc_x', 'l_foot_acc_y', 'l_foot_acc_z', 'r_foot_acc_x', 'r_foot_acc_y', 'r_foot_acc_z'
                    # , 'trunk_gyr_x', 'trunk_gyr_y', 'trunk_gyr_z', 'pelvis_gyr_x', 'pelvis_gyr_y', 'pelvis_gyr_z',
                    # 'l_thigh_gyr_x', 'l_thigh_gyr_y', 'l_thigh_gyr_z',
                    # 'r_thigh_gyr_x', 'r_thigh_gyr_y', 'r_thigh_gyr_z',
                    # 'l_shank_gyr_x', 'l_shank_gyr_y', 'l_shank_gyr_z',
                    # 'r_shank_gyr_x', 'r_shank_gyr_y', 'r_shank_gyr_z',
                    # 'l_foot_gyr_x', 'l_foot_gyr_y', 'l_foot_gyr_z', 'r_foot_gyr_x', 'r_foot_gyr_y', 'r_foot_gyr_z'
                    ]

# get training data
for gait_name in training_gait_names:
    gait_file_path = training_data_path + 'GaitData\gait_' + gait_name + '.csv'
    gait_data = pd.read_csv(gait_file_path)
    gait_data_len = gait_data.shape[0]

    # initialize training y
    l_knee_l_training = np.concatenate((l_knee_l_training, gait_data.as_matrix(
        columns=['l_knee_l_x', 'l_knee_l_y', 'l_knee_l_z'])))
    l_knee_r_training = np.concatenate((l_knee_r_training, gait_data.as_matrix(
        columns=['l_knee_r_x', 'l_knee_r_y', 'l_knee_r_z'])))
    r_knee_l_training = np.concatenate((r_knee_l_training, gait_data.as_matrix(
        columns=['r_knee_l_x', 'r_knee_l_y', 'r_knee_l_z'])))
    r_knee_r_training = np.concatenate((r_knee_r_training, gait_data.as_matrix(
        columns=['r_knee_r_x', 'r_knee_r_y', 'r_knee_r_z'])))
    l_cop_training = np.concatenate((l_cop_training, gait_data.as_matrix(
        columns=['c_1_x', 'c_1_y', 'c_1_z'])))
    r_cop_training = np.concatenate((r_cop_training, gait_data.as_matrix(
        columns=['c_2_x', 'c_2_y', 'c_2_z'])))
    l_force_training = np.concatenate((l_force_training, gait_data.as_matrix(
        columns=['f_1_x', 'f_1_y', 'f_1_z'])))
    r_force_training = np.concatenate((r_force_training, gait_data.as_matrix(
        columns=['f_2_x', 'f_2_y', 'f_2_z'])))

    # initialize training x
    acc_training = np.concatenate((acc_training, gait_data.as_matrix(columns=acc_column_names)))


# get testing data
for gait_name in testing_gait_names:
    gait_file_path = testing_data_path + 'GaitData\gait_' + gait_name + '.csv'
    gait_data = pd.read_csv(gait_file_path)
    gait_data_len = gait_data.shape[0]

    # initialize testing y
    l_knee_l_testing = np.concatenate((l_knee_l_testing, gait_data.as_matrix(
        columns=['l_knee_l_x', 'l_knee_l_y', 'l_knee_l_z'])))
    l_knee_r_testing = np.concatenate((l_knee_r_testing, gait_data.as_matrix(
        columns=['l_knee_r_x', 'l_knee_r_y', 'l_knee_r_z'])))
    r_knee_l_testing = np.concatenate((r_knee_l_testing, gait_data.as_matrix(
        columns=['r_knee_l_x', 'r_knee_l_y', 'r_knee_l_z'])))
    r_knee_r_testing = np.concatenate((r_knee_r_testing, gait_data.as_matrix(
        columns=['r_knee_r_x', 'r_knee_r_y', 'r_knee_r_z'])))
    l_cop_testing = np.concatenate((l_cop_testing, gait_data.as_matrix(
        columns=['c_1_x', 'c_1_y', 'c_1_z'])))
    r_cop_testing = np.concatenate((r_cop_testing, gait_data.as_matrix(
        columns=['c_2_x', 'c_2_y', 'c_2_z'])))
    l_force_testing = np.concatenate((l_force_testing, gait_data.as_matrix(
        columns=['f_1_x', 'f_1_y', 'f_1_z'])))
    r_force_testing = np.concatenate((r_force_testing, gait_data.as_matrix(
        columns=['f_2_x', 'f_2_y', 'f_2_z'])))

    # initialize testing x
    acc_testing = np.concatenate((acc_testing, gait_data.as_matrix(columns=acc_column_names)))


# as for knee_side, 0 represents left knee, 1 represents right knee
# as for data_set, 0 for training, 1 for testing
dynamic_var[subject_name + '_training' + '_left'].set_data(l_knee_l_training, l_knee_r_training,
                                                           l_force_training, l_cop_training)
dynamic_var[subject_name + '_training' + '_right'].set_data(r_knee_l_training, r_knee_r_training,
                                                            r_force_training, r_cop_training)
dynamic_var[subject_name + '_testing' + '_left'].set_data(l_knee_l_testing, l_knee_r_testing,
                                                          l_force_testing, l_cop_testing)
dynamic_var[subject_name + '_testing' + '_right'].set_data(r_knee_l_testing, r_knee_r_testing,
                                                           r_force_testing, r_cop_testing)

KAM_data_l_training = dynamic_var[subject_name + '_training' + '_left'].get_KAM()
KAM_data_r_training = - dynamic_var[subject_name + '_training' + '_right'].get_KAM()
KAM_data_l_testing = dynamic_var[subject_name + '_testing' + '_left'].get_KAM()
KAM_data_r_testing = - dynamic_var[subject_name + '_testing' + '_right'].get_KAM()
KAM_data_training = np.column_stack([KAM_data_l_training, KAM_data_r_training])
KAM_data_testing = np.column_stack([KAM_data_l_testing, KAM_data_r_testing])


# plt.figure()
# plt.plot(KAM_data_l_training)  # check the KAM data
# plt.figure()
# plt.plot(KAM_data_l_testing)  # check the KAM data
# plt.show()


def method_evaluation_both(model):


    model.fit(acc_training, KAM_data_training)
    score = model.score(acc_testing, KAM_data_testing)
    result = model.predict(acc_testing)
    plt.figure()
    plt.plot(KAM_data_testing[:, 0], 'b', label='true value')
    plt.plot(result[:, 0], 'r', label='predicted value')
    plt.title('R2: %f' % score)
    plt.legend()
    plt.figure()
    plt.plot(KAM_data_testing[:, 1], 'b', label='true value')
    plt.plot(result[:, 1], 'r', label='predicted value')
    plt.title('R2: %f' % score)
    plt.legend()

# from sklearn import svm
# model_SVR = svm.SVR()
# method_evaluation_both(model_SVR)

# from sklearn import neighbors
# model_KNN = neighbors.KNeighborsRegressor()
# method_evaluation_both(model_KNN)

from sklearn import ensemble
model_random_forest = ensemble.RandomForestRegressor()
method_evaluation_both(model_random_forest)

# # check the acc data
# plt.figure()
# plt.plot(acc_training[:,  23])

plt.show()
