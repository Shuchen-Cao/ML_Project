function xsens_raw_data = get_xsens_raw_data(folder_path, sensor_num, xsens_trial_num, xsens_export_num)

xsens_delimiter = '\t';
xsens_names = get_xsens_names();
for iXsens = 1: sensor_num
    eval(strcat('file_sub_name = xsens_names.s', num2str(iXsens), ';'));
    file_name = strcat(file_sub_name, xsens_trial_num, xsens_export_num, '.txt');
    file_path = strcat(folder_path, file_name);
    % s1 represents for sensor 1
    eval(['xsens_raw_data.s', num2str(iXsens), ' = dlmread(file_path, xsens_delimiter, 7, 0);']);
end

end