from scipy.signal import butter, filtfilt

from csvprocessor import find_trajectory_start_row, get_data_names
from GaitDataClass import GaitData
import scipy.io as sio
import xlrd
import numpy as np
import pandas as pd

file_date = '20171229\\'  # different subject are named according to their experiment date
raw_data_path = 'D:\Tian\Research\Projects\ML Project\data\\' + file_date  # path of raw vicon data
processed_data_path = 'D:\Tian\Research\Projects\ML Project\processed_data\\' + file_date  # path of processed data
save_path = processed_data_path + 'GaitData\\'  # the processed data will be saved in this folder

# gait names in the sequence of the experiment
if file_date == '20171229\\':
    gait_names = (
        'normal', 'toeout', 'toein', 'largeSW', 'largeTS', 'toein_largeSW', 'toeout_largeSW', 'change_location')
else:
    gait_names = ('normal', 'toeout', 'toein', 'largeSW', 'largeTS', 'toein_largeSW', 'toeout_largeSW')
# gait_names = ('normal', 'toeout', 'toein', 'largeSW', 'largeTS', 'toein_largeSW', 'toeout_largeSW')

gait_num = gait_names.__len__()  # number of total gaits

sync_file_path = raw_data_path + 'vicon_motion_sync.xlsx'  # the file path which contains sync information
sync_file = xlrd.open_workbook(sync_file_path)  # open the file
sync_sheet = sync_file.sheets()[1]
sync_vicon_raw1 = sync_sheet.row_values(5)  # the vicon start frame of each gait is stored in the 5th column
sync_vicon_raw2 = sync_sheet.row_values(6)  # the vicon end frame of each gait is stored in the 6th column
sync_vicon = np.matrix([sync_vicon_raw1[2:2 + gait_num], sync_vicon_raw2[2:2 + gait_num]])
sync_vicon += -1  # try to set offset
sync_vicon = sync_vicon.astype(int)  # transfer the start frame as int

# get offset
offset_plate1 = sio.loadmat(processed_data_path + 'offset_plate1.mat')  # load mat file
offset_plate1 = offset_plate1['offset_plate1'][0]  # extract data from the mat file
offset_plate2 = sio.loadmat(processed_data_path + 'offset_plate2.mat')  # load mat file
offset_plate2 = offset_plate2['offset_plate2'][0]  # extract data from the mat file

# set filter parameters
filter_order = 4
wn_force = 50 / (1000 / 2)  # unified frequency
b_force_plate, a_force_plate = butter(filter_order, wn_force, btype='low')
wn_marker = 6 / 50
b_marker, a_marker = butter(filter_order, wn_marker, btype='low')

dynamic_var = locals()  # dynamically name the Object and csv file
i_gait = 0  # increase 1 at the end of the loop
for gait_name in gait_names:
    object_name = 'gait_' + gait_name  # the name of the GaitData object and file
    dynamic_var[object_name] = GaitData(object_name)  # initial the object
    # xsens data
    file_path_xsens = processed_data_path + 'xsens\\' + gait_names[i_gait] + '.mat'  # xsens file folder
    data = sio.loadmat(file_path_xsens)  # load mat file
    data_xsens = data['data']  # extract data from the mat file
    xsens_num = data_xsens.shape[2]  # get the number of xsens sensors

    # vicon data
    file_path_vicon = raw_data_path + 'Trial0' + str(i_gait) + '.csv'  # the path of vicon raw data
    start_row = sync_vicon[0, i_gait]  # start row of this gait
    end_row = sync_vicon[1, i_gait]  # end row of this gaitof trajectory info

    force_data_raw = pd.read_csv(file_path_vicon, skiprows=[0, 1, 2, 4], nrows=10 * (end_row + 3))
    force_data_raw = force_data_raw.as_matrix(  # only get useful columns
        columns=['Frame', 'Fx', 'Fy', 'Fz', 'Cx', 'Cy', 'Cz', 'Fx.1', 'Fy.1', 'Fz.1', 'Cx.1', 'Cy.1', 'Cz.1'])

    force_data_raw[:, 4] += offset_plate1[0]
    force_data_raw[:, 5] += offset_plate1[1]
    force_data_raw[:, 6] += offset_plate1[2]
    force_data_raw[:, 10] += offset_plate2[0]
    force_data_raw[:, 11] += offset_plate2[1]
    force_data_raw[:, 12] += offset_plate2[2]

    # filter the force data
    force_data = force_data_raw[:, 1:]
    force_data = filtfilt(b_force_plate, a_force_plate, force_data, axis=0)  # filtering
    force_data = np.column_stack((force_data_raw[:, 0], force_data))
    force_data_range = range(start_row * 10 - 1, end_row * 10, 10)
    force_data = force_data[force_data_range]

    # trajectory info are at the end of vicon export file
    marker_offset = find_trajectory_start_row(file_path_vicon)  # get the offset
    skip_range = list(range(0, marker_offset + 1))  # skip the force data and a couple rows when selecting marker data
    marker_data_raw = pd.read_csv(file_path_vicon, skiprows=skip_range)
    marker_data_raw = marker_data_raw.as_matrix()

    # use low pass filter to filter marker data
    marker_data = marker_data_raw[:, 2:]  # Frame column does not need a filter
    marker_data = filtfilt(b_marker, a_marker, marker_data, axis=0)  # filtering
    marker_data = np.column_stack([marker_data_raw[:, 0], marker_data])
    marker_data_range = range(start_row, end_row + 1)
    marker_data = marker_data[marker_data_range]

    # combine all the xsens, force plate and marker data
    data_raw = np.column_stack([force_data - 1, marker_data])  # stack the Frame column
    for i_xsens in range(xsens_num):  # stack xsens data
        data_raw = np.column_stack([data_raw, data_xsens[:, :, i_xsens]])

    data = pd.DataFrame(data_raw)  # transfer matrix to DataFrame
    data_column_names = get_data_names(file_path_vicon, marker_offset - 1)  # get standard column names
    data.columns = data_column_names  # change column names to the standard name
    dynamic_var[object_name].set_data(data)  # pass the data to the matrix
    # save as csv
    dynamic_var[object_name].save_as_csv(save_path)  # save as csv
    dynamic_var[object_name].clear_old_csv(save_path, object_name)  # delete the former file
    i_gait += 1  # prepare to process next gait data
