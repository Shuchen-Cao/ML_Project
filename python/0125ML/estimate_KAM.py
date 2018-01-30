import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from KAMDataClass import KAMData
import scipy.io as sio
from scipy.signal import butter, filtfilt
import csv

file_date = '20171229\\'
processed_data_path = 'D:\Tian\Research\Projects\ML Project\processed_data\\' + file_date
# gait names in the sequence of the experiment
if file_date == '20171229\\':
    gait_names = (
    'normal', 'toeout', 'toein', 'largeSW', 'largeTS', 'toein_largeSW', 'toeout_largeSW', 'change_location')
else:
    gait_names = ('normal', 'toeout', 'toein', 'largeSW', 'largeTS', 'toein_largeSW', 'toeout_largeSW')

knee_marker_names = ('l_knee_l_x', 'l_knee_l_y', 'l_knee_l_z')

gait_num = 7
# validation_set_len = 10000  # the length of validation
test_set_len = 500  # the length of data for test

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


l_offset = sio.loadmat(processed_data_path + 'offset_plate1.mat')  # load mat file
l_offset = l_offset['offset_plate1'][0]  # extract data from the mat file

r_offset = sio.loadmat(processed_data_path + 'offset_plate2.mat')  # load mat file
r_offset = r_offset['offset_plate2'][0]  # extract data from the mat file

for gait_name in gait_names:
    gait_file_path = processed_data_path + 'GaitData\gait_' + gait_name + '.csv'
    gait_data = pd.read_csv(gait_file_path)

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




# use low pass filter to filter marker data
filter_order = 4
wn_marker = 6 / 50
b_marker, a_marker = butter(filter_order, wn_marker, btype='low')
l_knee_l_testing = filtfilt(b_marker, a_marker, l_knee_l_testing, axis=0)
l_knee_r_testing = filtfilt(b_marker, a_marker, l_knee_r_testing, axis=0)

# modification of offset
# l_offset[0] = -l_offset[0]
# r_offset[0] = -r_offset[0]
# l_offset[0] += 20
# r_offset[0] += -20

l_cop_training += l_offset
r_cop_training += r_offset
l_cop_testing += l_offset
r_cop_testing += r_offset

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

# reshape data so that sklearn could be used
# KAM_data_l_training.reshape([KAM_data_l_training.shape[0], 1])
# KAM_data_r_training.reshape((KAM_data_r_training.shape[0], 1))
# KAM_data_l_testing.reshape(KAM_data_l_testing.shape[0], 1)
# KAM_data_r_testing.reshape(KAM_data_r_testing.shape[0], 1)




# plt.figure()
# plt.plot(KAM_data_l)  # check the KAM data
# plt.figure()
# plt.plot(KAM_data_r)  # check the KAM data
# plt.show()


def method_evaluation(model):
    model.fit(acc_training, KAM_data_l_training)
    score = model.score(acc_testing, KAM_data_l_testing)
    result = model.predict(acc_testing)
    plt.figure()
    plt.plot(KAM_data_l_testing, 'b', label='true value')
    plt.plot(result, 'r', label='predicted value')
    plt.title('score: %f' % score)
    plt.legend()

#
# from sklearn import tree
# model_decision_tree = tree.DecisionTreeRegressor()
# method_evaluation(model_decision_tree)
#
# from sklearn import svm
# model_SVR = svm.SVR()
# method_evaluation(model_SVR)

# from sklearn import neighbors
# model_KNN = neighbors.KNeighborsRegressor()
# method_evaluation(model_KNN)

from sklearn import ensemble
model_random_forest = ensemble.RandomForestRegressor()
method_evaluation(model_random_forest)

plt.show()












