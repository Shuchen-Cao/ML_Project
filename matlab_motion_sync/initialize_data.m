% this file is used to check hitting ground point and get the walking period
clc
close all

% the path of the raw data
date = '20171228\';
folder_path = strcat('D:\Tian\Research\Projects\ML Project\data\', date);
% the path of the processed xsens data
processed_path = strcat('D:\Tian\Research\Projects\ML Project\processed_data\', date);
sensor_num = 8;  % the number of sensor
gait_num = 7;  % the number of gaits

% the file which stores sync info
sync_file = strcat(folder_path, 'vicon_motion_sync.xlsx');
% get the sync data
[~, ~, sync_raw_data] = xlsread(sync_file, 1);
xsens_file_names = get_xsens_file_names();
gait_names = get_gait_names();

for i_gait = 1:gait_num
    % get vicon 
    frame_vicon = cell2mat(sync_raw_data(2:9, i_gait+2));  % get sync frame to calculate frame difference
    frame_xsens = cell2mat(sync_raw_data(12:19, i_gait+2));
    frame_diff_raw = frame_vicon - frame_xsens;  % frame difference of 8 hits
    frame_median = median(frame_diff_raw);  % get the median value of frame difference data
    
    %% this part could be omitted
    % error if some threshold are largely different from others
    for i_stamp = 1:8  % 跺脚
        error_value =  abs(frame_diff_raw(i_stamp) - frame_median);
        % fprintf(strcat(num2str(error_value), ','));
        if error_value > 4  % threshold of sync error
            error('error in sync frames');  
        end
    end
    variance_value = frame_median - mean(frame_diff_raw);
    
    % error if the median and mean separated from each other too much
    if variance_value > 1  % threshold of sync variantion
        error('separated too much')
    end
    if rem(frame_median, 1) == 0
        frame_diff = frame_median;
    else
        frame_diff = round(mean(frame_diff_raw));
    end
    gait_name = eval(['gait_names.g', num2str(i_gait)]);
    file_left_foot = strcat(folder_path, gait_name, xsens_file_names.s7);
    file_right_foot = strcat(folder_path, gait_name, xsens_file_names.s8);
    
    data_left_foot = mtbFileLoader(file_left_foot);
    data_right_foot = mtbFileLoader(file_right_foot);
    
    % 1, 2 are start and end, 3, 4 are the detected point
    xsens_interval_current = get_walking_interval(data_left_foot, frame_xsens(4));
    xsens_interval(:, i_gait) = xsens_interval_current;
    
%     figure
%     scrsz = get(0,'ScreenSize');
%     set(gcf,'Position',scrsz);
%     % check the cut point
%     subplot(2, 1, 1);
%     plot(data_left_foot(:, 4)); hold on
%     plot(xsens_interval_current(1), data_left_foot(xsens_interval_current(1), 4), '*');
%     plot(xsens_interval_current(2), data_left_foot(xsens_interval_current(2), 4), '*');
%     plot(xsens_interval_current(3), data_left_foot(xsens_interval_current(3), 4), '*');
%     plot(xsens_interval_current(4), data_left_foot(xsens_interval_current(4), 4), '*');
%     subplot(2, 1, 2);
%     plot(data_right_foot(:, 4)); hold on
%     plot(xsens_interval_current(1), data_right_foot(xsens_interval_current(1), 4), '*');
%     plot(xsens_interval_current(2), data_right_foot(xsens_interval_current(2), 4), '*');
%     plot(xsens_interval_current(3), data_right_foot(xsens_interval_current(3), 4), '*');
%     plot(xsens_interval_current(4), data_right_foot(xsens_interval_current(4), 4), '*');

    % 6 for acc, gyr and 8 for 8 sensors
    data = zeros(xsens_interval_current(2) - xsens_interval_current(1) + 1, 6, 8);
    for i_sensor = 1:sensor_num - 2  % 6 xsenses except left and right foot
        xsens_name = eval(['xsens_file_names.s', num2str(i_sensor)]);
        file_name = strcat(folder_path, gait_name, xsens_name);
        
        data_uncut = mtbFileLoader(file_name);
        data_cut = data_uncut(xsens_interval_current(1):xsens_interval_current(2), 2:7);
        data(:, :, i_sensor) = data_cut;
    end
    % left and right foot do not have to be read again
    data_cut = data_left_foot(xsens_interval_current(1):xsens_interval_current(2), 2:7);
    data(:, :, 7) = data_cut;
    data_cut = data_right_foot(xsens_interval_current(1):xsens_interval_current(2), 2:7);
    data(:, :, 8) = data_cut;
    
    gait_file = eval(['gait_names.g', num2str(i_gait)]);
    gait_file = split(gait_file, '\');
    gait_file = strcat(processed_path, 'xsens\', char(gait_file(1)), '.mat');
    save(gait_file, 'data');
    vicon_interval(:, i_gait) = xsens_interval_current(1:2) + frame_diff;
end

% write the sync data to the excel file
xlrange_vicon = 'C6:J7';
xlswrite(sync_file, vicon_interval, 2, xlrange_vicon);
xlrange_xsens = 'C2:J3';
xlswrite(sync_file, xsens_interval(1:2, :), 2, xlrange_xsens);

beep();




function interval = get_walking_interval(data_left_foot, start_frame)
check_interval = 30;  % 每隔此区间检验方差
std_thd = 0.08;  % 0.16 for 20171228, 0.08 for 20171229
for i_start = start_frame + 400:check_interval:size(data_left_foot, 1)
    fluct = std(data_left_foot(i_start:i_start + check_interval, 4));  % fluctuation
    if fluct > std_thd
        break;
    end
end
margin_start = 450;  % 开头多剪掉的部分
margin_end = 700;  % 结尾多剪掉的部分
interval(1) = i_start + margin_start + check_interval / 2;

for i_end = i_start + margin_start:check_interval:size(data_left_foot, 1)
    fluct = std(data_left_foot(i_end:i_end + check_interval, 4));  % fluctuation
    if fluct < std_thd
        break;
    end
end
interval(2) = i_end - margin_end;
interval(3) = i_start;
interval(4) = i_end;
interval = interval';
end





