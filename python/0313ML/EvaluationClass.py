
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from keras.models import *


class Evaluation:

    def __init__(self, x_training, x_testing, y_training, y_testing, params_column_names,
                 gait_num, sub_num, x_scalar=[], y_scalar=[]):
        self.__x_training = x_training
        self.__x_testing = x_testing
        self.__y_training = y_training
        self.__y_testing = y_testing
        self.__gait_num = gait_num
        self.__sub_num = sub_num
        self.__train_set_len = int(self.__x_training.shape[0] / self.__gait_num / self.__sub_num)
        self.__test_set_len = int(self.__x_testing.shape[0] / self.__gait_num / self.__sub_num)
        self.__params_column_names = params_column_names
        self.__do_scaling = False
        if x_scalar:
            self.__x_scalar = x_scalar
            if y_scalar:
                self.__y_scalar = y_scalar
                self.__do_scaling = True

        if self.__do_scaling:
            self.__x_training = self.__x_scalar.transform(self.__x_training)
            self.__x_testing = self.__x_scalar.transform(self.__x_testing)
            self.__y_training = self.__y_scalar.transform(self.__y_training)
            self.__y_testing = self.__y_scalar.transform(self.__y_testing)
        else:  # transfer dataframe to ndarray
            self.__x_training = self.__x_training.as_matrix()
            self.__x_testing = self.__x_testing.as_matrix()
            self.__y_training = self.__y_training.as_matrix()
            self.__y_testing = self.__y_testing.as_matrix()

    def shuffle(self):
        # the seed should be the same for two shuffles
        np.random.seed(20)
        np.random.shuffle(self.__x_training)
        np.random.seed(20)
        np.random.shuffle(self.__y_training)
        # self.__y_training = self.__y_training.sample(frac=1, random_state=10)

    def x_3D_transform(self, window_len=10, skip_len=2):
        ori_train_len = self.__x_training.shape[0]
        trans_train_len = int((ori_train_len - window_len + skip_len) / skip_len)
        x_training_3D = np.zeros([trans_train_len, window_len, self.__x_training.shape[1]])
        y_training_3D = np.zeros([trans_train_len, self.__y_training.shape[1]])
        for i_sample in range(0, trans_train_len):
            x_training_3D[i_sample, :, :] = self.__x_training[i_sample*skip_len:i_sample*skip_len+window_len, :]
            y_training_3D[i_sample, :] = self.__y_training[i_sample*skip_len + window_len - 1]
        self.__x_training = x_training_3D
        self.__y_training = y_training_3D

        ori_train_len = self.__x_testing.shape[0]
        trans_train_len = int((ori_train_len - window_len + skip_len) / skip_len)
        x_testing_3D = np.zeros([trans_train_len, window_len, self.__x_testing.shape[1]])
        y_testing_3D = np.zeros([trans_train_len, self.__y_testing.shape[1]])
        for i_sample in range(0, trans_train_len):
            x_testing_3D[i_sample, :, :] = self.__x_testing[i_sample*skip_len:i_sample*skip_len+window_len, :]
            y_testing_3D[i_sample, :] = self.__y_testing[i_sample*skip_len + window_len - 1]
        self.__x_testing = x_testing_3D
        self.__y_testing = y_testing_3D

        # change the __test_set_len so that the plot can be properly shown
        self.__test_set_len = int(self.__test_set_len / skip_len)

    def time_series_weight(self):
        trans_matrice = np.matrix([[0.2, 0, 0, 0, 0],
                                   [0, 0.4, 0, 0, 0],
                                   [0, 0, 0.6, 0, 0],
                                   [0, 0, 0, 0.8, 0],
                                   [0, 0, 0, 0, 1]])
        for i_slice in range(self.__x_training.shape[0]):
            self.__x_training[i_slice, :, :] = trans_matrice * self.__x_training[i_slice, :, :]
        for i_slice in range(self.__x_testing.shape[0]):
            self.__x_testing[i_slice, :, :] = trans_matrice * self.__x_testing[i_slice, :, :]

    def evaluate_nn(self, model):
        batch_size = 50  # the size of data that be trained together
        # lr = learning rate, the other params are default values
        optimizer = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        # val_loss = validation loss, patience is the tolerance
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        # epochs is the maximum training round, validation split is the size of the validation set,
        # callback stops the training if the validation was not approved
        model.fit(self.__x_training, self.__y_training, batch_size=batch_size,
                  epochs=100, validation_split=0.2, callbacks=[early_stopping])
        result = model.predict(self.__x_testing, batch_size=batch_size)

        if self.__do_scaling:
            self.__y_testing = self.__y_scalar.inverse_transform(self.__y_testing)
            result = self.__y_scalar.inverse_transform(result)

        score = r2_score(self.__y_testing, result, multioutput='raw_values')
        for i_plot in range(result.shape[1]):
            plt.figure()
            plt.plot(self.__y_testing[:, i_plot], 'b', label='true value')
            plt.plot(result[:, i_plot], 'r', label='predicted value')
            plt.title(self.__params_column_names[i_plot] + '  R2: ' + str(score[i_plot])[0:5])
            plt.legend()
            for i_subject in range(0, self.__sub_num):
                for i_gait in range(1, self.__gait_num):
                    line_x_gait = self.__test_set_len * i_gait + i_subject * (self.__test_set_len * self.__gait_num)
                    plt.plot((line_x_gait, line_x_gait), (-0.5, 0.5), 'y--')
                if i_subject != 0:
                    line_x_sub = self.__x_testing.shape[0] / self.__sub_num * i_subject
                    plt.plot((line_x_sub, line_x_sub), (-0.5, 0.5), 'black')
        plt.show()

    def evaluate_sklearn(self, model):
        model.fit(self.__x_training.as_matrix(), self.__y_training.as_matrix())
        result = model.predict(self.__x_testing.as_matrix())
        score = r2_score(self.__y_testing.as_matrix(), result, multioutput='raw_values')
        if self.__do_scaling:
            self.__y_testing = self.__y_scalar.inverse_transform(self.__y_testing)
            result = self.__y_scalar.inverse_transform(result)
        # plot
        for i_plot in range(result.shape[1]):
            plt.figure()
            plt.plot(self.__y_testing.as_matrix()[:, i_plot], 'b', label='true value')
            plt.plot(result[:, i_plot], 'r', label='predicted value')
            plt.title(self.__params_column_names[i_plot] + '  R2: ' + str(score[i_plot])[0:5])
            plt.legend()
            for i_subject in range(0, self.__sub_num):
                for i_gait in range(1, self.__gait_num):
                    line_x_gait = self.__test_set_len * i_gait + i_subject * (self.__test_set_len * self.__gait_num)
                    plt.plot((line_x_gait, line_x_gait), (-0.5, 0.5), 'y--')
                if i_subject != 0:
                    line_x_sub = self.__x_testing.shape[0] / self.__sub_num * i_subject
                    plt.plot((line_x_sub, line_x_sub), (-0.5, 0.5), 'black')
        plt.show()
        # plt.savefig()


