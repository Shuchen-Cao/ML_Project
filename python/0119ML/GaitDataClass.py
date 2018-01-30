import os


class GaitData(object):
    def __init__(self, name):
        self.__len = 0
        self.__gait_name = name

    def get_gait_name(self):
        return self.__gait_name

    def set_data(self, data):
        self.__data = data
        self.__len = data.shape[0]

    def get_data(self):
        return self.__data

    def get_len(self):
        return self.__len

    def save_as_csv(self, path):
        file_path = path + self.__gait_name + '.csv'
        i_file = 0
        while os.path.isfile(file_path):
            i_file += 1
            file_path = path + self.__gait_name + str(i_file) + '.csv'
        self.__data.to_csv(file_path)

    @staticmethod
    def clear_old_csv(path, file_name):
        i_file = 1
        previous_file_path = path + file_name + '.csv'
        file_path = path + file_name + str(i_file) + '.csv'
        while os.path.isfile(file_path):
            i_file += 1
            os.remove(previous_file_path)
            previous_file_path = file_path
            file_path = path + file_name + str(i_file) + '.csv'
        os.rename(previous_file_path, path + file_name + '.csv')
