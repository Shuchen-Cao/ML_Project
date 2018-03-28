close all

subject_num = 2;
gait_num = 7;
folder_path = 'D:\Tian\Research\Projects\ML Project\processed_data\';
data_len_training = 15000;
data_len_testing = 500;

% acc_training = zeros(data_len_training * subject_num * gait_num, 24);
% param_training = zeros(data_len_training * subject_num * gait_num, 2);
% acc_testing = zeros(data_len_testing * subject_num * gait_num, 24);
% param_testing = zeros(data_len_testing * subject_num * gait_num, 2);
acc_training = [];
param_training = [];
acc_testing = [];
param_testing = [];

for i_sub = 1:subject_num
    date = get_date(i_sub);
    folder_gait = strcat(folder_path, date, 'GaitData\');
    folder_param = strcat(folder_path, date, 'GaitParameters\');
    for i_gait = 1:gait_num
        name = get_gait_names(i_gait);
        file_path_gait = strcat(folder_gait, 'gait_', name, '.csv');
        file_path_param = strcat(folder_param, 'params_of_', name, '.csv');
%         index_training = 
%         index_testing = 
        new_acc_training = csvread(file_path_gait, 1, 153, [1, 153, data_len_training, 155]);
        new_acc_training = [new_acc_training, csvread(file_path_gait, 1, 159, [1, 159, data_len_training, 161])];
        new_acc_training = [new_acc_training, csvread(file_path_gait, 1, 165, [1, 165, data_len_training, 167])];
        new_acc_training = [new_acc_training, csvread(file_path_gait, 1, 171, [1, 171, data_len_training, 173])];
        new_acc_training = [new_acc_training, csvread(file_path_gait, 1, 177, [1, 177, data_len_training, 179])];
        new_acc_training = [new_acc_training, csvread(file_path_gait, 1, 183, [1, 183, data_len_training, 185])];
        new_acc_training = [new_acc_training, csvread(file_path_gait, 1, 189, [1, 189, data_len_training, 191])];
        new_acc_training = [new_acc_training, csvread(file_path_gait, 1, 195, [1, 195, data_len_training, 197])];
        acc_training = [acc_training; new_acc_training];
        
        new_acc_testing = csvread(file_path_gait, data_len_training, 153, [data_len_training, 153, data_len_training + data_len_testing - 1, 155]);
        new_acc_testing = [new_acc_testing, csvread(file_path_gait, data_len_training, 159, [data_len_training, 159, data_len_training + data_len_testing - 1, 161])];
        new_acc_testing = [new_acc_testing, csvread(file_path_gait, data_len_training, 165, [data_len_training, 165, data_len_training + data_len_testing - 1, 167])];
        new_acc_testing = [new_acc_testing, csvread(file_path_gait, data_len_training, 171, [data_len_training, 171, data_len_training + data_len_testing - 1, 173])];
        new_acc_testing = [new_acc_testing, csvread(file_path_gait, data_len_training, 177, [data_len_training, 177, data_len_training + data_len_testing - 1, 179])];
        new_acc_testing = [new_acc_testing, csvread(file_path_gait, data_len_training, 183, [data_len_training, 183, data_len_training + data_len_testing - 1, 185])];
        new_acc_testing = [new_acc_testing, csvread(file_path_gait, data_len_training, 189, [data_len_training, 189, data_len_training + data_len_testing - 1, 191])];
        new_acc_testing = [new_acc_testing, csvread(file_path_gait, data_len_training, 195, [data_len_training, 195, data_len_training + data_len_testing - 1, 197])];
        acc_testing = [acc_testing; new_acc_testing];
        
        param_training = [param_training; csvread(file_path_param, 1, 13, [1, 13, data_len_training, 14])];
        param_testing = [param_testing; csvread(file_path_param, data_len_training, 13, [data_len_training, 13, data_len_training + data_len_testing - 1, 14])];
        
    end
end
save acc_training acc_training
save acc_testing acc_testing
save param_training param_training
save param_testing param_testing
