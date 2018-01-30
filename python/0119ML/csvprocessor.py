import csv


def find_trajectory_start_row(file):
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        i_row = 0
        for row in reader:
            if row and (row[0] == 'Trajectories'):
                offset = i_row + 3
                break
            i_row += 1
    return offset


def get_marker_names(file, row_num):
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # get the row of names
        for i_row, rows in enumerate(reader):
            if i_row == row_num:
                the_row = rows
                break
    names_raw = list()

    # get the names and put them in a list
    for name in the_row:
        if name != '':
            names_raw.append(name)

    # bulid a new
    names = list()
    for name in names_raw:
        name = name.split(':')[1]
        names.append(name + '_x')
        names.append(name + '_y')
        names.append(name + '_z')
    names.insert(0, 'marker_frame')
    return names


def get_xsens_names():
    names = list(['trunk_acc_x', 'trunk_acc_y', 'trunk_acc_z', 'trunk_gyr_x', 'trunk_gyr_y', 'trunk_gyr_z',
                  'pelvis_acc_x', 'pelvis_acc_y', 'pelvis_acc_z', 'pelvis_gyr_x', 'pelvis_gyr_y', 'pelvis_gyr_z',
                  'l_thigh_acc_x', 'l_thigh_acc_y', 'l_thigh_acc_z', 'l_thigh_gyr_x', 'l_thigh_gyr_y', 'l_thigh_gyr_z',
                  'r_thigh_acc_x', 'r_thigh_acc_y', 'r_thigh_acc_z', 'r_thigh_gyr_x', 'r_thigh_gyr_y', 'r_thigh_gyr_z',
                  'l_shank_acc_x', 'l_shank_acc_y', 'l_shank_acc_z', 'l_shank_gyr_x', 'l_shank_gyr_y', 'l_shank_gyr_z',
                  'r_shank_acc_x', 'r_shank_acc_y', 'r_shank_acc_z', 'r_shank_gyr_x', 'r_shank_gyr_y', 'r_shank_gyr_z',
                  'l_foot_acc_x', 'l_foot_acc_y', 'l_foot_acc_z', 'l_foot_gyr_x', 'l_foot_gyr_y', 'l_foot_gyr_z',
                  'r_foot_acc_x', 'r_foot_acc_y', 'r_foot_acc_z', 'r_foot_gyr_x', 'r_foot_gyr_y', 'r_foot_gyr_z'])
    return names


def get_force_names():
    names = list(['force_frame', 'f_1_x', 'f_1_y', 'f_1_z', 'c_1_x', 'c_1_y', 'c_1_z',
                  'f_2_x', 'f_2_y', 'f_2_z', 'c_2_x', 'c_2_y', 'c_2_z'])
    return names


def get_data_names(file, marker_name_row_num):
    names = get_force_names() + get_marker_names(file, marker_name_row_num) + get_xsens_names()
    return names

