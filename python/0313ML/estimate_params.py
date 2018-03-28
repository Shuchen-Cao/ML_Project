# two subjects were combined


import pandas as pd
from sklearn import preprocessing
from EvaluationClass import Evaluation
from keras.models import Sequential
from keras.layers import *

# file_dates = ['20171228\\']
file_dates = ['20171228\\', '20171229\\']


skip_set_len = 0  # skiping wrong data
training_set_len = 1500  # should be around 15000
test_set_len = 1000  # the length of data for test

# gait names in the sequence of the experiment
gait_names = ('normal', 'toeout', 'toein', 'largeSW', 'largeTS',
              'toein_largeSW', 'toeout_largeSW')
# gait_names = ('normal', 'toeout', 'toein', 'largeSW', 'toein_largeSW', 'toeout_largeSW')

# initialize x
acc_column_names = [
    'trunk_acc_x', 'trunk_acc_y', 'trunk_acc_z', 'pelvis_acc_x', 'pelvis_acc_y', 'pelvis_acc_z',
    'l_thigh_acc_x', 'l_thigh_acc_y', 'l_thigh_acc_z',
    'r_thigh_acc_x', 'r_thigh_acc_y', 'r_thigh_acc_z',
    'l_shank_acc_x', 'l_shank_acc_y', 'l_shank_acc_z',
    'r_shank_acc_x', 'r_shank_acc_y', 'r_shank_acc_z',
    'l_foot_acc_x', 'l_foot_acc_y', 'l_foot_acc_z', 'r_foot_acc_x', 'r_foot_acc_y', 'r_foot_acc_z',
    'trunk_gyr_x', 'trunk_gyr_y', 'trunk_gyr_z', 'pelvis_gyr_x', 'pelvis_gyr_y', 'pelvis_gyr_z',
    'l_thigh_gyr_x', 'l_thigh_gyr_y', 'l_thigh_gyr_z',
    'r_thigh_gyr_x', 'r_thigh_gyr_y', 'r_thigh_gyr_z',
    'l_shank_gyr_x', 'l_shank_gyr_y', 'l_shank_gyr_z',
    'r_shank_gyr_x', 'r_shank_gyr_y', 'r_shank_gyr_z',
    'l_foot_gyr_x', 'l_foot_gyr_y', 'l_foot_gyr_z', 'r_foot_gyr_x', 'r_foot_gyr_y', 'r_foot_gyr_z'
]

# initialize y
params_column_names = [
    # 'f_1_x', 'f_1_y',
    # 'f_1_z',
    # 'f_2_x', 'f_2_y',
    # 'f_2_z',
    # 'c_1_x', 'c_1_y', 'c_2_x', 'c_2_y',
    'KAM_l', 'KAM_R',
    # 'strike_l', 'strike_r', 'off_l', 'off_r',
    # 'step_width', 'FPA_l', 'FPA_r',
    # 'pelvis_angle_x', 'pelvis_angle_y', 'pelvis_angle_z',
    # 'trunk_swag',
    # 'ankle_flexion_angle_l', 'ankle_flexion_angle_r',
    # 'knee_flexion_angle_l', 'knee_flexion_angle_r',
    # 'hip_flexion_angle_l', 'hip_flexion_angle_r'
]

acc_training = pd.DataFrame()
acc_testing = pd.DataFrame()
params_training = pd.DataFrame()
params_testing = pd.DataFrame()

for date in file_dates:
    data_path = 'D:\Tian\Research\Projects\ML Project\processed_data\\' + date
    for gait_name in gait_names:
        # get x
        gait_file_path = data_path + 'GaitData\gait_' + gait_name + '.csv'
        gait_data = pd.read_csv(gait_file_path)
        gait_data = pd.read_csv(gait_file_path)[acc_column_names]
        acc_training = acc_training.append(gait_data[0:training_set_len], ignore_index=True)
        acc_testing = acc_testing.append(
            gait_data[skip_set_len + training_set_len:skip_set_len + training_set_len + test_set_len],
            ignore_index=True)

        # get y
        params_file_path = data_path + 'GaitParameters\params_of_' + gait_name + '.csv'
        params_data = pd.read_csv(params_file_path)[params_column_names]
        params_training = params_training.append(params_data[0:training_set_len], ignore_index=True)
        params_testing = params_testing.append(
            params_data[skip_set_len + training_set_len:skip_set_len + training_set_len + test_set_len],
            ignore_index=True)


# # Standard scaler
acc_std_scaler = preprocessing.StandardScaler().fit(acc_training)
param_std_scaler = preprocessing.StandardScaler().fit(params_training)

# # Max min scaler
# min_max_scaler = preprocessing.MinMaxScaler().fit(acc_training)

# # Robust scaler
# acc_robust_scaler = preprocessing.RobustScaler().fit(acc_training)

# param_robust_scaler = preprocessing.RobustScaler().fit(params_training)

# # Regularization
# regulizer = preprocessing.Normalizer().fit(acc_training)

# without scalar
# my_evaluation = Evaluation(acc_training, acc_testing, params_training, params_testing, params_column_names,
#                            gait_names.__len__(), file_dates.__len__())

# with scalar
my_evaluation = Evaluation(acc_training, acc_testing, params_training, params_testing, params_column_names,
                           gait_names.__len__(), file_dates.__len__(), acc_std_scaler, param_std_scaler)

## learning algorithms
# from sklearn import svm
# model_SVR = svm.SVR()
# method_evaluation_both(model_SVR)

# from sklearn import neighbors
# model_KNN = neighbors.KNeighborsRegressor(n_neighbors=12)
# my_evaluation.evaluate_sklearn(model_KNN)

# from sklearn import ensemble
# model_random_forest = ensemble.RandomForestRegressor()
# method_evaluation_both(model_random_forest)


# # bp nn
# my_evaluation.shuffle()
# model_bp = Sequential()
# model_bp.add(Dense(120, input_dim=acc_column_names.__len__(), activation='relu'))
# model_bp.add(Dense(80, activation='relu'))
# model_bp.add(Dense(8, activation='relu'))
# # model_bp.add(Dropout(0.5))
# model_bp.add(Dense(params_column_names.__len__(), input_dim=48, activation='linear'))
# my_evaluation.evaluate_nn(model_bp)


# # average bp
# acc_training = np.column_stack([1*acc_training[0:-4], 1 * acc_training[1:-3], 1 * acc_training[2:-2],
#                                 1 * acc_training[3:-1], 1 * acc_training[4:]])
# acc_testing = np.column_stack([1 * acc_testing[0:-4], 1 * acc_testing[1:-3], 1 * acc_testing[2:-2],
#                                1 * acc_testing[3:-1], 1 * acc_testing[4:]])
# acc_std_scaler = preprocessing.StandardScaler().fit(acc_training)
# param_std_scaler = preprocessing.StandardScaler().fit(params_training)
# params_training = params_training[4:]
# params_testing = params_testing[4:]
# my_evaluation = Evaluation(acc_training, acc_testing, params_training, params_testing, params_column_names,
#                            gait_names.__len__(), file_dates.__len__(), acc_std_scaler, param_std_scaler)
# my_evaluation.shuffle()
# model_bp = Sequential()
# model_bp.add(Dense(200, input_dim=acc_column_names.__len__() * 5, activation='relu'))
# model_bp.add(Dense(120, activation='relu'))
# model_bp.add(Dense(20, activation='relu'))
# # model_bp.add(Dropout(0.5))
# model_bp.add(Dense(params_column_names.__len__(), input_dim=48, activation='linear'))
# my_evaluation.evaluate_nn(model_bp)


# # CNN
# window_len = 10
# skip_len = 2
# feature_len = 5
# my_evaluation.x_3D_transform(window_len, skip_len)
# my_evaluation.shuffle()
# model_cnn = Sequential()
# model_cnn.add(Conv1D(10, feature_len, activation='relu', input_shape=(window_len, acc_column_names.__len__())))
# model_cnn.add(Flatten())
# model_cnn.add(Dense(180, activation='relu'))
# model_cnn.add(Dense(80, activation='relu'))
# model_cnn.add(Dense(20, activation='relu'))
# model_cnn.add(Dense(params_column_names.__len__(), activation='linear'))
# my_evaluation.evaluate_nn(model_cnn)


# # simple RNN
# my_evaluation.x_3D_transform(1, 1)
# my_evaluation.shuffle()
# model_rnn = Sequential()
# regularizer = regularizers.l2(0.001)
# model_rnn.add(SimpleRNN(140, input_dim=acc_column_names.__len__(), activation='relu', kernel_regularizer=regularizer))
# model_rnn.add(Dense(75, activation='relu'))
# model_rnn.add(Dense(20, activation='relu'))
# model_rnn.add(Dense(params_column_names.__len__(), activation='linear'))
# model_rnn.add(Activation('linear'))
# my_evaluation.evaluate_nn(model_rnn)

# # LSTM
# my_evaluation.x_3D_transform(1, 1)
# my_evaluation.shuffle()
# model_lstm = Sequential()
# # regularizer = regularizers.l2(0.001)
# model_lstm.add(LSTM(50, input_dim=acc_column_names.__len__(), activation='relu'))
# model_lstm.add(Dense(100, activation='relu'))
# model_lstm.add(Dense(20, activation='relu'))
# model_lstm.add(Dense(params_column_names.__len__(), activation='linear'))
# model_lstm.add(Activation('linear'))
# my_evaluation.evaluate_nn(model_lstm)





























