
test=c3dserver();
needFilter = 0;
plate = 2;
date = '20171229\';
raw_data_path = strcat('D:\Tian\Research\Projects\ML Project\data\', date, 'calibration\');
processed_data_path = strcat('D:\Tian\Research\Projects\ML Project\processed_data\', date);
%% Trial04
openc3d(test,1,strcat(raw_data_path, 'plate', num2str(plate), '.c3d'));
rawData.Trial01_Force = csvread(strcat(raw_data_path, 'plate', num2str(plate),'.CSV'),5,0);
rawData.Trial01_Components = get3dtargets(test,0);
 
D = (rawData.Trial01_Components.DL + rawData.Trial01_Components.DR)/2;
P = rawData.Trial01_Components.DL - rawData.Trial01_Components.ML;
Cop_Vicon = P+D;
if plate == 1
    Cop_Force = rawData.Trial01_Force(:,3:5);% forceplate1
elseif plate == 2
    Cop_Force = rawData.Trial01_Force(:,6:8);% forceplate2
end
    
if needFilter==1
    [b,a]=butter(2,20/(1000/2));
    Cop_Force = filter(b,a,Cop_Force);
end

Filtered_Force = zeros(length(Cop_Vicon), 3);
for i = 1:length(Cop_Vicon)
    Filtered_Force(i, :) = median(Cop_Force(10*(i - 1) + 1: 10*i, :));
end

Different = Filtered_Force - Cop_Vicon;
Offset.mean = mean(Different(100:end,:));
Offset.std = std(Different(100:end,:));

% if plate == 1
%     offset_plate1 = Offset.mean;
%     save(strcat(processed_data_path, 'offset_plate1'), 'offset_plate1')
% elseif plate == 2
%     offset_plate2 = Offset.mean;
%     save(strcat(processed_data_path, 'offset_plate2'), 'offset_plate2')
% end




