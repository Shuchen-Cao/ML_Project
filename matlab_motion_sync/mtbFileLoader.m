function data = mtbFileLoader(filename)
%% open the log file
Fs = 100;
h = actxserver('xsensdeviceapi_com64.IXsensDeviceApi');
h.XsControl_openLogFile(filename);
deviceID = cell2mat(h.XsControl_mainDeviceIds());
device = h.XsControl_device(deviceID);
h.XsDevice_setOptions(device, h.XsOption_XSO_RetainRecordingData, 0);

%% Load the log file and wait until it is loaded
h.registerevent({'onProgressUpdated', @eventhandlerXsens});
h.XsDevice_loadLogFile(device);
fileLoaded = 0;
while  fileLoaded == 0
    % wait untill maxSamples are arrived
    pause(.2)
end
fprintf('\n File fully loaded\n')

%% start data extracting
% get total number of samples
nSamples = h.XsDevice_getDataPacketCount(device);
% 7 columns in the data are time, 3 acc, 3 gyr
data = zeros(nSamples, 7);
hasPacketCounter = false;
hasTimeData = false;
hasSdiData = false;
readSample = 0;
% for loop to extract the data
for iSample = 0:nSamples
    % get data packet
    dataPacket =  h.XsDevice_getDataPacketByIndex(device,iSample);
    % check if dataPacket is a data packet
    if dataPacket
        readSample = readSample+1;
        % see if data packet contains certain data
        if h.XsDataPacket_containsSdiData(dataPacket)
            hasSdiData = true;
            % extract data, data will always be in cells
            sdiData = cell2mat(h.XsDataPacket_sdiData(dataPacket));
            data(readSample, 2:4) = sdiData(5:7) * 100;
            raw_gyr = sdiData(2:4);
            n = norm(raw_gyr);     % 取模
            if n>eps        % 设置阈值eps，小于该阈值则视为0。
                data(readSample, 5:7) = (2*asin(n)*Fs)*(raw_gyr/n);
            else
                data(readSample, 5:7) = zeros(3,1);
            end
        end
        data(readSample, 1) = h.XsDataPacket_sampleTimeFine(dataPacket);
    end
end

%% close port and object
h.XsControl_close();

delete(h); % release COM-object
clear h;

%% event handling
    function eventhandlerXsens(varargin)
        % device pointer is zero for progressUpdated
        devicePtr = varargin{3}{1};
        % The current progress
        currentProgress = varargin{3}{2};
        % The total work to be done
        total = varargin{3}{3};
        % Identifier, in the case the file name which is loaded
        identifier = varargin{3}{4};
        if currentProgress == 100
            fileLoaded = 1;
        end
    end


end