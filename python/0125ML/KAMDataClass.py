# this class is used to store the data used to calculate KAM
# the units of weight, height are Kg, m respectively

import numpy as np
from numpy.core.umath_tests import inner1d
from numpy.linalg import norm

class KAMData(object):
    def __init__(self, weight, height, name='unnamed'):
        self.__weight = weight
        self.__height = height
        self.__name = name

    def set_data(self, knee_l, knee_r, force, cop, data_set=0, knee_side=0):
        # as for knee_side, 0 represents left knee, 1 represents right knee
        # as for data_set, 0 for training, 1 for testing

        # find the vector vertical to knee vector
        knee_center = (knee_l + knee_r) / 2
        knee_vector = knee_l - knee_r
        knee_vector[:, 2] = 0  # only need the data in xy plane
        data_len = knee_vector.shape[0]
        x_vector = np.zeros([data_len, 3])
        x_vector[:, 1] = 1  # set each column as [1 0 0]

        # easy to prove that dot(vertical_vector, knee_vector) = 0
        vertical_vector = x_vector - knee_vector * \
            (inner1d(x_vector, knee_vector) / inner1d(knee_vector, knee_vector))[:, None]
        vertical_vector = vertical_vector / norm(vertical_vector, axis=1)[:, None]
        force_arm = (knee_center - cop) / 1000  # mm to m
        KAM_raw = np.cross(force_arm, force)
        KAM = inner1d(KAM_raw, vertical_vector) / (self.__weight * self.__height)
        if data_set == 0:
            if knee_side == 0:
                self.__l_KAM_training = -KAM
            else:
                self.__r_KAM_training = KAM
        elif data_set == 1:
            if knee_side == 0:
                self.__l_KAM_testing = -KAM
            else:
                self.__r_KAM_testing = KAM

    def get_KAM(self, data_set=0, knee_side=0):
        # as for knee_side, 0 represents left knee, 1 represents right knee
        # as for data_set, 0 for training, 1 for testing

        # check if the data have been initialized
        try:
            if data_set == 0:
                if knee_side == 0:
                    return self.__l_KAM_training
                else:
                    return self.__r_KAM_training
            elif data_set == 1:
                if knee_side == 0:
                    return self.__l_KAM_testing
                else:
                    return self.__r_KAM_testing
        except Exception:
            print('The data have not been initialized. 0 will be returned.')
            return 0

