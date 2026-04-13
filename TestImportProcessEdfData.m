clear; clc;

% ===== paths =====
repoDir      = '/Users/takamanabe/Documents/Git/NeuralDataPreprocessing';
fieldtripDir = '/Users/takamanabe/Library/CloudStorage/GoogleDrive-mahiro.original@gmail.com/My Drive/NU_Research/MS/BMD_ENG499/Code/MATLAB_Plugin/fieldtrip-20260408';
edfRoot      = '/Volumes/PHI_Epilepsy/epilepsyNeuralData/clinicalDataProspective/P002/natus/edf';
partStr      = 'P002';

addpath(genpath(repoDir));
addpath(fieldtripDir);
ft_defaults;

% ログ未初期化で writePreprocessingLogs が落ちるのを防ぐ
tmpLogDir = fullfile(tempdir,'edf_guard_logs');
if ~exist(tmpLogDir,'dir'); mkdir(tmpLogDir); end
writePreprocessingLogs(2, ['Create logs for CGuardBatch in ' tmpLogDir]);
cleanupObj = onCleanup(@() writePreprocessingLogs(3,[])); %#ok<NASGU>

% lookup
ep = defineEpilepsyProcessingParameters();
[~, channelNamesAll, ~] = readPreprocessingLookupTable(ep,false);
channelNamesTable = channelNamesAll.(partStr);

% EDF一覧（1次ディレクトリのみ）
edfFiles = dir(fullfile(edfRoot,'*.edf'));
n = numel(edfFiles);
if n == 0
    error('No EDF files found under: %s', edfRoot);
end

results = table('Size',[n 11], ...
    'VariableTypes', {'string','string','string','double','double','double','double','string','string','double','string'}, ...
    'VariableNames', {'file','status','errorType','lenChannelNames','maxChannelCount','allC_all','allC_1N','errorMessage','errorFile','errorLine','errorFunc'});

for i = 1:n
    f = fullfile(edfFiles(i).folder, edfFiles(i).name);

    % 初期値
    status = "SETUP_ERROR";
    lenChannelNames = NaN;
    maxChannelCount = NaN;
    allC_all = NaN;
    allC_1N = NaN;
    errType = "";
    errMsg = "";
    errFile = "";
    errLine = NaN;
    errFunc = "";

    try
        hdr = ft_read_header(f);

        if strcmpi(hdr.label{1},'Event')
            firstNeuralChannel = 2;
        else
            firstNeuralChannel = 1;
        end

        numberNeuralChannels = sum(~isnan(channelNamesTable.natusCount));
        lastNeuralChannel = numberNeuralChannels + (firstNeuralChannel - 1);
        lastNeuralChannel = min(lastNeuralChannel, numel(hdr.label));

        channelNames = hdr.label(firstNeuralChannel:lastNeuralChannel);
        rmv = contains(lower(channelNames),'dc') | contains(lower(channelNames),'ground');
        channelNames(rmv) = [];

        lenChannelNames = length(channelNames);
        maxChannelCount = channelNamesTable.natusCount(end);
        allC_all = double(all(contains(channelNames,'C')));

        if maxChannelCount <= lenChannelNames
            allC_1N = double(all(contains(channelNames(1:maxChannelCount),'C')));
        end

        % checkEdfChannelNamesVsChannelTable だけ最小実行
        channelDataStub = zeros(lenChannelNames,2);
        headerStub = struct('label',{channelNames});

        try
            checkEdfChannelNamesVsChannelTable(channelNames,channelNamesTable,channelDataStub,headerStub);
            status = "OK_NO_ERROR";
            errType = "NONE";
        catch ME
            status = "ERROR";
            errMsg = string(ME.message);
            msgLower = lower(errMsg);
            if contains(msgLower, "all channels have c in their name so how is there less than 128 or 256?")
                errType = "C_GUARD_ALL";
            elseif contains(msgLower, "all channels that should be real have c in their name?")
                errType = "C_GUARD_TAIL";
            elseif contains(msgLower, "index exceeds array bounds")
                errType = "INDEX_BOUNDS";
            else
                errType = "OTHER_ERROR";
            end
            if ~isempty(ME.stack)
                errFile = string(ME.stack(1).file);
                errLine = ME.stack(1).line;
                errFunc = string(ME.stack(1).name);
            end
        end

    catch ME
        status = "SETUP_ERROR";
        errType = "SETUP_ERROR";
        errMsg = string(ME.message);
        if ~isempty(ME.stack)
            errFile = string(ME.stack(1).file);
            errLine = ME.stack(1).line;
            errFunc = string(ME.stack(1).name);
        end
    end

    results.file(i) = string(edfFiles(i).name);
    results.status(i) = status;
    results.errorType(i) = errType;
    results.lenChannelNames(i) = lenChannelNames;
    results.maxChannelCount(i) = maxChannelCount;
    results.allC_all(i) = allC_all;
    results.allC_1N(i) = allC_1N;
    results.errorMessage(i) = errMsg;
    results.errorFile(i) = errFile;
    results.errorLine(i) = errLine;
    results.errorFunc(i) = errFunc;

    fprintf('[%d/%d] %s -> %s (%s)\n', i, n, edfFiles(i).name, status, errType);
end

disp(results(:,{'file','status','errorType','lenChannelNames','maxChannelCount','allC_all','allC_1N','errorLine','errorFunc','errorMessage'}));
[errorTypes, errorCounts] = groupcounts(results.errorType);
disp(table(errorTypes,errorCounts));

% 必要なら保存
outCsv = fullfile(tmpLogDir, sprintf('%s_checkEdfChannelNamesVsChannelTable_results.csv', partStr));
writetable(results, outCsv);
fprintf('Saved: %s\n', outCsv);
