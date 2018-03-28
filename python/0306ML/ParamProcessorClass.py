import pandas as pd
import numpy as np
from numpy.core.umath_tests import inner1d
from numpy.linalg import norm
import matplotlib.pyplot as plt


class ParamProcessor:
    def __init__(self, path, weight, height):
        self.__gait_data = pd.read_csv(path)
        self.__weight = weight
        self.__height = height

    def get_KAM(self):
        l_knee_l = self.__gait_data.as_matrix(columns=['l_knee_l_x', 'l_knee_l_y', 'l_knee_l_z'])
        l_knee_r = self.__gait_data.as_matrix(columns=['l_knee_r_x', 'l_knee_r_y', 'l_knee_r_z'])
        r_knee_l = self.__gait_data.as_matrix(columns=['r_knee_l_x', 'r_knee_l_y', 'r_knee_l_z'])
        r_knee_r = self.__gait_data.as_matrix(columns=['r_knee_r_x', 'r_knee_r_y', 'r_knee_r_z'])
        l_force = self.__gait_data.as_matrix(columns=['f_1_x', 'f_1_y', 'f_1_z'])
        r_force = self.__gait_data.as_matrix(columns=['f_2_x', 'f_2_y', 'f_2_z'])
        l_cop = self.__gait_data.as_matrix(columns=['c_1_x', 'c_1_y', 'c_1_z'])
        r_cop = self.__gait_data.as_matrix(columns=['c_2_x', 'c_2_y', 'c_2_z'])

        # the sign of left KAM should be turned around
        left_KAM = - self.__calculate_KAM(l_knee_l, l_knee_r, l_force, l_cop)
        right_KAM = self.__calculate_KAM(r_knee_l, r_knee_r, r_force, r_cop)
        return np.column_stack([left_KAM, right_KAM])

    def __calculate_KAM(self, knee_l, knee_r, force, cop):
        # find the vector vertical to knee vector
        knee_center = (knee_l + knee_r) / 2
        knee_vector = knee_l - knee_r
        knee_vector[:, 2] = 0  # only need the data in xy plane
        data_len = knee_vector.shape[0]
        x_vector = np.zeros([data_len, 3])
        x_vector[:, 1] = 1  # set each column as [1 0 0]

        # easy to prove that dot(vertical_vector, knee_vector) = 0
        vertical_vector = x_vector - knee_vector * (inner1d(x_vector, knee_vector) /
                                                    inner1d(knee_vector, knee_vector))[:, None]
        vertical_vector = vertical_vector / norm(vertical_vector, axis=1)[:, None]
        force_arm = (knee_center - cop) / 1000  # mm to m
        KAM_raw = np.cross(force_arm, force)
        KAM = inner1d(KAM_raw, vertical_vector) / (self.__weight * self.__height)
        return KAM

    def check_KAM(self, data_KAM):
        plt.figure()
        plt.plot(data_KAM)
        plt.title('KAM')

    def get_COP(self):
        l_cop = self.__gait_data.as_matrix(columns=['c_1_x', 'c_1_y', 'c_1_z'])
        r_cop = self.__gait_data.as_matrix(columns=['c_2_x', 'c_2_y', 'c_2_z'])
        return np.column_stack([l_cop, r_cop])

    def get_force(self):
        l_force = self.__gait_data.as_matrix(columns=['f_1_x', 'f_1_y', 'f_1_z'])
        r_force = self.__gait_data.as_matrix(columns=['f_2_x', 'f_2_y', 'f_2_z'])
        return np.column_stack([l_force, r_force])

    def get_heel_strike_event(self, threshold=20):
        l_force = self.__gait_data.as_matrix(columns=['f_1_x', 'f_1_y', 'f_1_z'])
        l_force = norm(l_force, axis=1)
        data_len = l_force.shape[0]
        l_strikes = np.zeros(data_len, dtype=np.int8)
        comparison_len = 2
        for i_point in range(comparison_len - 1, data_len - comparison_len):
            if l_force[i_point - 2] > threshold:
                continue
            if l_force[i_point - 1] > threshold:
                continue
            if l_force[i_point] < threshold:
                continue
            if l_force[i_point + 1] < threshold:
                continue
            if l_force[i_point + 2] < threshold:
                continue
            l_strikes[i_point - 1] = 1

        r_force = self.__gait_data.as_matrix(columns=['f_2_x', 'f_2_y', 'f_2_z'])
        r_force = norm(r_force, axis=1)
        data_len = r_force.shape[0]
        r_strikes = np.zeros(data_len, dtype=np.int8)
        comparison_len = 2
        for i_point in range(comparison_len - 1, data_len - comparison_len):
            if r_force[i_point - 2] > threshold:
                continue
            if r_force[i_point - 1] > threshold:
                continue
            if r_force[i_point] < threshold:
                continue
            if r_force[i_point + 1] < threshold:
                continue
            if r_force[i_point + 2] < threshold:
                continue
            r_strikes[i_point - 1] = 1

        return np.column_stack([l_strikes, r_strikes])

    def get_toe_off_event(self, threshold=20):
        l_force = self.__gait_data.as_matrix(columns=['f_1_x', 'f_1_y', 'f_1_z'])
        l_force = norm(l_force, axis=1)
        data_len = l_force.shape[0]
        l_off = np.zeros(data_len, dtype=np.int8)
        comparison_len = 2
        for i_point in range(comparison_len - 1, data_len - comparison_len):
            if l_force[i_point - 2] < threshold:
                continue
            if l_force[i_point - 1] < threshold:
                continue
            if l_force[i_point] > threshold:
                continue
            if l_force[i_point + 1] > threshold:
                continue
            if l_force[i_point + 2] > threshold:
                continue
            l_off[i_point] = 1

        r_force = self.__gait_data.as_matrix(columns=['f_2_x', 'f_2_y', 'f_2_z'])
        r_force = norm(r_force, axis=1)
        data_len = r_force.shape[0]
        r_off = np.zeros(data_len, dtype=np.int8)
        comparison_len = 2
        for i_point in range(comparison_len - 1, data_len - comparison_len):
            if r_force[i_point - 2] < threshold:
                continue
            if r_force[i_point - 1] < threshold:
                continue
            if r_force[i_point] > threshold:
                continue
            if r_force[i_point + 1] > threshold:
                continue
            if r_force[i_point + 2] > threshold:
                continue
            r_off[i_point] = 1

        return np.column_stack([l_off, r_off])

    def check_strikes_off(self, strikes, offs, check_len=5000):
        forces = self.get_force()
        plt.figure()
        plt.plot(norm(forces[0:check_len, 0:2], axis=1))
        for i in range(0, check_len):
            if strikes[i, 0] == 1:
                plt.plot(i, 1.7, '.', color='red')
            if offs[i, 0] == 1:
                plt.plot(i, 1.7, '.', color='yellow')
        plt.legend()
        plt.title('heel strike & toe off')

    def get_trunk_swag(self):
        C7 = self.__gait_data.as_matrix(columns=['C7_x', 'C7_y', 'C7_z'])
        l_PSIS = self.__gait_data.as_matrix(columns=['l_PSIS_x', 'l_PSIS_y', 'l_PSIS_z'])
        r_PSIS = self.__gait_data.as_matrix(columns=['r_PSIS_x', 'r_PSIS_y', 'r_PSIS_z'])
        middle_PSIS = (l_PSIS + r_PSIS) / 2
        vertical_vector = C7 - middle_PSIS
        return - 180 / np.pi * np.arctan(vertical_vector[:, 0] / vertical_vector[:, 2])

    @staticmethod
    def check_trunk_swag(self, data_trunk_swag):
        plt.figure()
        plt.plot(data_trunk_swag)
        plt.title('trunk swag')

    def get_step_width(self):
        l_ankle_l = self.__gait_data.as_matrix(columns=['l_ankle_l_x', 'l_ankle_l_y', 'l_ankle_l_z'])
        l_ankle_r = self.__gait_data.as_matrix(columns=['l_ankle_r_x', 'l_ankle_r_y', 'l_ankle_r_z'])
        r_ankle_l = self.__gait_data.as_matrix(columns=['r_ankle_l_x', 'r_ankle_l_y', 'r_ankle_l_z'])
        r_ankle_r = self.__gait_data.as_matrix(columns=['r_ankle_r_x', 'r_ankle_r_y', 'r_ankle_r_z'])
        data_len = l_ankle_l.shape[0]
        step_width = np.zeros(data_len)
        heel_strikes = self.get_heel_strike_event()
        # set the right feet as dominate feet
        new_step = False
        for i_point in range(0, data_len):
            # check left foot
            if heel_strikes[i_point, 0] == 1:
                ankle_l = (l_ankle_l[i_point, 0] + l_ankle_r[i_point, 0]) / 2
                new_step = True
            # check right foot
            if heel_strikes[i_point, 1] == 1:
                if new_step:
                    ankle_r = (r_ankle_l[i_point, 0] + r_ankle_r[i_point, 0]) / 2
                    step_width[i_point] = ankle_r - ankle_l
                    new_step = False

        return step_width

    def check_step_width(self, step_widths, check_len=5000):
        forces = self.get_force()
        plt.figure()
        plt.plot(norm(forces[0:check_len, 3:5], axis=1))
        for i in range(0, check_len):
            if step_widths[i] != 0:
                plt.plot(i, 1.7, '.', color='red')
        plt.legend()
        plt.title('step width')

    def get_FPA(self):
        l_toe_mt2 = self.__gait_data.as_matrix(columns=['l_toe_mt2_x', 'l_toe_mt2_y', 'l_toe_mt2_z'])
        l_cal = self.__gait_data.as_matrix(columns=['l_cal_x', 'l_cal_y', 'l_cal_z'])
        data_len = l_toe_mt2.shape[0]
        left_FPAs = np.zeros(data_len)
        heel_strikes = self.get_heel_strike_event()
        for i_point in range(0, data_len):
            if heel_strikes[i_point, 0] == 1:
                forward_vector = l_toe_mt2[i_point, :] - l_cal[i_point, :]
                left_FPAs[i_point] = - 180 / np.pi * np.arctan(forward_vector[0] / forward_vector[1])

        r_toe_mt2 = self.__gait_data.as_matrix(columns=['r_toe_mt2_x', 'r_toe_mt2_y', 'r_toe_mt2_z'])
        r_cal = self.__gait_data.as_matrix(columns=['r_cal_x', 'r_cal_y', 'r_cal_z'])
        right_FPAs = np.zeros(data_len)
        for i_point in range(0, data_len):
            if heel_strikes[i_point, 1] == 1:
                forward_vector = r_toe_mt2[i_point, :] - r_cal[i_point, :]
                right_FPAs[i_point] = 180 / np.pi * np.arctan(forward_vector[0] / forward_vector[1])

        return np.column_stack([left_FPAs, right_FPAs])

    def check_FPA(self, FPAs, check_len=5000):
        forces = self.get_force()
        plt.figure()
        plt.plot(norm(forces[0:check_len, 0:2], axis=1))
        for i in range(0, check_len):
            if FPAs[i, 0] != 0:
                plt.plot(i, 1.7, '.', color='red')
        plt.legend()
        plt.title('FPA')

    # be careful about absent values
    def get_pelvis_angle(self):
        l_PSIS = self.__gait_data.as_matrix(columns=['l_PSIS_x', 'l_PSIS_y', 'l_PSIS_z'])
        r_PSIS = self.__gait_data.as_matrix(columns=['r_PSIS_x', 'r_PSIS_y', 'r_PSIS_z'])
        l_ASIS = self.__gait_data.as_matrix(columns=['l_ASIS_x', 'l_ASIS_y', 'l_ASIS_z'])
        r_ASIS = self.__gait_data.as_matrix(columns=['r_ASIS_x', 'r_ASIS_y', 'r_ASIS_z'])

        # # posterior side offset
        # l_PSIS[:, 2] = l_PSIS[:, 2] - 35
        # r_PSIS[:, 2] = r_PSIS[:, 2] - 35

        x_vector = r_ASIS - l_ASIS
        y_vector = (l_ASIS + r_ASIS) / 2 - (l_PSIS + r_PSIS) / 2
        x_vector_norm = x_vector / norm(x_vector, axis=1)[:, None]
        y_vector_norm = y_vector / norm(y_vector, axis=1)[:, None]
        z_vector_norm = np.cross(x_vector_norm, y_vector_norm)
        alpha = 180 / np.pi * np.arctan2(-z_vector_norm[:, 1], z_vector_norm[:, 2])
        beta = 180 / np.pi * np.arcsin(z_vector_norm[:, 0])
        gamma = 180 / np.pi * np.arctan(-y_vector_norm[:, 0], x_vector_norm[:, 0])
        return np.column_stack([alpha, beta, gamma])

    @staticmethod
    def check_pelvis_angle(self, pelvis_angles):
        plt.figure()
        plt.plot(pelvis_angles)
        plt.title('pelvis angle')

    def get_knee_flexion_angle(self):
        l_knee_l = self.__gait_data.as_matrix(columns=['l_knee_l_x', 'l_knee_l_y', 'l_knee_l_z'])
        l_ankle_l = self.__gait_data.as_matrix(columns=['l_ankle_l_x', 'l_ankle_l_y', 'l_ankle_l_z'])
        l_hip = self.__gait_data.as_matrix(columns=['l_hip_x', 'l_hip_y', 'l_hip_z'])
        shank_vector = l_ankle_l[:, 1:] - l_knee_l[:, 1:]
        thigh_vector = l_hip[:, 1:] - l_knee_l[:, 1:]
        l_knee_angles = 180 - self.__law_of_cosines(shank_vector, thigh_vector)

        r_knee_r = self.__gait_data.as_matrix(columns=['r_knee_r_x', 'r_knee_r_y', 'r_knee_r_z'])
        r_ankle_r = self.__gait_data.as_matrix(columns=['r_ankle_r_x', 'r_ankle_r_y', 'r_ankle_r_z'])
        r_hip = self.__gait_data.as_matrix(columns=['r_hip_x', 'r_hip_y', 'r_hip_z'])
        shank_vector = r_ankle_r[:, 1:] - r_knee_r[:, 1:]
        thigh_vector = r_hip[:, 1:] - r_knee_r[:, 1:]
        r_knee_angles = 180 - self.__law_of_cosines(shank_vector, thigh_vector)

        return np.column_stack([l_knee_angles, r_knee_angles])

    def check_knee_flexion_angle(self, knee_flexion_angle):
        plt.figure()
        plt.plot(knee_flexion_angle)
        plt.title('knee flexion angle')

    def get_ankle_flexion_angle(self):
        l_knee_l = self.__gait_data.as_matrix(columns=['l_knee_l_x', 'l_knee_l_y', 'l_knee_l_z'])
        l_knee_r = self.__gait_data.as_matrix(columns=['l_knee_r_x', 'l_knee_r_y', 'l_knee_r_z'])
        l_ankle_l = self.__gait_data.as_matrix(columns=['l_ankle_l_x', 'l_ankle_l_y', 'l_ankle_l_z'])
        l_ankle_r = self.__gait_data.as_matrix(columns=['l_ankle_r_x', 'l_ankle_r_y', 'l_ankle_r_z'])
        l_toe_mt2 = self.__gait_data.as_matrix(columns=['l_toe_mt2_x', 'l_toe_mt2_y', 'l_toe_mt2_z'])
        l_cal = self.__gait_data.as_matrix(columns=['l_cal_x', 'l_cal_y', 'l_cal_z'])
        l_knee_center = (l_knee_l[:, 1:] + l_knee_r[:, 1:]) / 2
        l_ankle_center = (l_ankle_l[:, 1:] + l_ankle_r[:, 1:]) / 2
        l_shank_vector = l_ankle_center - l_knee_center
        l_foot_vector = l_toe_mt2[:, 1:] - l_cal[:, 1:]
        l_ankle_angles = self.__law_of_cosines(l_shank_vector, l_foot_vector) - 90

        r_knee_l = self.__gait_data.as_matrix(columns=['r_knee_l_x', 'r_knee_l_y', 'r_knee_l_z'])
        r_knee_r = self.__gait_data.as_matrix(columns=['r_knee_r_x', 'r_knee_r_y', 'r_knee_r_z'])
        r_ankle_l = self.__gait_data.as_matrix(columns=['r_ankle_l_x', 'r_ankle_l_y', 'r_ankle_l_z'])
        r_ankle_r = self.__gait_data.as_matrix(columns=['r_ankle_r_x', 'r_ankle_r_y', 'r_ankle_r_z'])
        r_toe_mt2 = self.__gait_data.as_matrix(columns=['r_toe_mt2_x', 'r_toe_mt2_y', 'r_toe_mt2_z'])
        r_cal = self.__gait_data.as_matrix(columns=['r_cal_x', 'r_cal_y', 'r_cal_z'])
        r_knee_center = (r_knee_l[:, 1:] + r_knee_r[:, 1:]) / 2
        r_ankle_center = (r_ankle_l[:, 1:] + r_ankle_r[:, 1:]) / 2
        r_shank_vector = r_ankle_center - r_knee_center
        r_foot_vector = r_toe_mt2[:, 1:] - r_cal[:, 1:]
        r_ankle_angles = self.__law_of_cosines(r_shank_vector, r_foot_vector) - 90

        return np.column_stack([l_ankle_angles, r_ankle_angles])

    def check_ankle_flexion_angle(self, ankle_flexion_angle):
        plt.figure()
        plt.plot(ankle_flexion_angle)
        plt.title('ankle flexion angle')

    @staticmethod
    def __law_of_cosines(vector1, vector2):
        vector3 = vector1 - vector2
        num = inner1d(vector1, vector1) + \
              inner1d(vector2, vector2) - inner1d(vector3, vector3)
        den = 2 * np.sqrt(inner1d(vector1, vector1)) * np.sqrt(inner1d(vector2, vector2))
        return 180 / np.pi * np.arccos(num / den)

    def get_hip_flexion_angle(self):
        l_PSIS = self.__gait_data.as_matrix(columns=['l_PSIS_x', 'l_PSIS_y', 'l_PSIS_z'])
        r_PSIS = self.__gait_data.as_matrix(columns=['r_PSIS_x', 'r_PSIS_y', 'r_PSIS_z'])
        C7 = self.__gait_data.as_matrix(columns=['C7_x', 'C7_y', 'C7_z'])
        middle_PSIS = (l_PSIS[:, 1:] + r_PSIS[:, 1:]) / 2
        vertical_vector = C7[:, 1:] - middle_PSIS

        l_knee_l = self.__gait_data.as_matrix(columns=['l_knee_l_x', 'l_knee_l_y', 'l_knee_l_z'])
        l_hip = self.__gait_data.as_matrix(columns=['l_hip_x', 'l_hip_y', 'l_hip_z'])
        thigh_vector = l_hip[:, 1:] - l_knee_l[:, 1:]
        l_hip_angle = 180 - self.__law_of_cosines(vertical_vector, thigh_vector)

        r_knee_r = self.__gait_data.as_matrix(columns=['r_knee_r_x', 'r_knee_r_y', 'r_knee_r_z'])
        r_hip = self.__gait_data.as_matrix(columns=['r_hip_x', 'r_hip_y', 'r_hip_z'])
        thigh_vector = r_hip[:, 1:] - r_knee_r[:, 1:]
        r_hip_angle = 180 - self.__law_of_cosines(vertical_vector, thigh_vector)

        return np.column_stack([l_hip_angle, r_hip_angle])

    def check_hip_flexion_angle(self, hip_flexion_angle):
        plt.figure()
        plt.plot(hip_flexion_angle)
        plt.title('hip flexion angle')

















