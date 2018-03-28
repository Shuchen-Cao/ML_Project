close all
load acc_training
load acc_testing
load param_training
load param_testing

acc_training = acc_training';
acc_testing = acc_testing';
param_training = param_training';
param_testing = param_testing';

% % 归一化
% [norm_input_train, input_ps] = mapminmax(input_train);
% [norm_output_train, output_ps] = mapminmax(output_train);

%构建BP神经网络
net = newff(acc_training, param_training, [12]);
net.trainParam.epochs = 200;
net.trainParam.lr = 0.1;
net.trainParam.goal = 1e-6;
%BP神经网络训练
net = train(net, acc_training, param_training);
% %测试样本归一化
% inputn_test = mapminmax('apply',input_test,input_ps);
%BP神经网络预测
save net net
BPoutput = sim(net, acc_testing);
% %%网络得到数据反归一化
% BPoutput = mapminmax('reverse', an ,output_ps);

for i_plot = 1:size(param_testing, 1)
    figure
    plot(param_testing(i_plot, :));
    hold on
    plot(BPoutput(i_plot, :));
end


% plot(data_KAM);
% hold on
% % [b, a] = butter(2, 0.2);        % 0.2为归一化后的截止频率，10Hz
% data_filtered_KAM = filter(b, a, data_KAM);
% plot(data_filtered_KAM);







