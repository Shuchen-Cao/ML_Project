% this function is used to get the xsens peaks

close all
clc

folder_trial = 'toeout\';  % the current processing trial
peak_offset = 0;  % 测力板取的是刚有力的点，因此xsens需要从峰值向左移动数格
% folder of the xsens raw data
folder_path = 'D:\Tian\Research\Projects\ML Project\data\20171229\';
sensor_names = get_xsens_file_names();  % the name of sensor

result = zeros(8, 1, 'uint16');  % the final result
scrsz = get(0,'ScreenSize');  % get the screen size
set(gcf,'Position',scrsz);  % set the figure size to the whole screen
plot_margin = 20;  % the margin of each plot
%% left foot
% the path of xsens file
xsens_file = strcat(folder_path, folder_trial, sensor_names.s7);
left_foot_data = mtbFileLoader(xsens_file);  % use the function to read xsens file
left_foot_data(1:200, :) = 0;  % not likely to happen in 2 seconds for now
% find acc_y peaks
[pks_raw, locs_raw] = findpeaks(left_foot_data(:, 4), 'MinPeakHeight', 30);

% % check if the left foot data is correct
% plot(left_foot_data(:, 4));hold on
% plot(locs_raw, pks_raw, '*');
% figure

% sort peaks and find four hitting ground peaks
[pks, locs] = sort_peak_raw(pks_raw, locs_raw);
subplot(2, 2, 1);  % plot the left start hit
% plot the hit imu data
plot(left_foot_data(locs(1) - plot_margin:locs(2) + plot_margin, 4)); hold on
% plot the selected peak
plot(locs(1:2) - locs(1) + plot_margin + 1, pks(1:2), '*')
title('start left');
subplot(2, 2, 2);  % plot the left end hit
% plot the hit imu data
plot(left_foot_data(locs(3) - plot_margin:locs(4) + plot_margin, 4)); hold on
% plot the selected peak
plot(locs(3:4) - locs(3) + plot_margin + 1, pks(3:4), '*')
title('end left');

result(1) = locs(1) - peak_offset;
result(3) = locs(2) - peak_offset;
result(5) = locs(3) - peak_offset;
result(7) = locs(4) - peak_offset;


%% right foot 
xsens_file = strcat(folder_path, folder_trial, sensor_names.s8);
right_foot_data = mtbFileLoader(xsens_file);
right_foot_data(1:200, :) = 0;
[pks_raw, locs_raw] = findpeaks(right_foot_data(:, 4),...
    'MinPeakHeight', 30);

[pks, locs] = sort_peak_raw(pks_raw, locs_raw);
subplot(2, 2, 3);  % plot the right end hit
% plot the hit imu data
plot(right_foot_data(locs(1) - plot_margin:locs(2) + plot_margin, 4)); hold on
% plot the selected peak
plot(locs(1:2) - locs(1) + plot_margin + 1, pks(1:2), '*')
title('start right');
subplot(2, 2, 4);  % plot the right end hit
% plot the hit imu data
plot(right_foot_data(locs(3) - plot_margin:locs(4) + plot_margin, 4)); hold on
% plot the selected peak
plot(locs(3:4) - locs(3) + plot_margin + 1, pks(3:4), '*')
title('end right');

result(2) = locs(1) - peak_offset;
result(4) = locs(2) - peak_offset;
result(6) = locs(3) - peak_offset;
result(8) = locs(4) - peak_offset;

% % check if the right foot data is correct
% figure
% plot(right_foot_data(:, 4));hold on
% plot(locs_raw, pks_raw, '*');




