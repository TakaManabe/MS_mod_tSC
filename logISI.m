function logISI(mainPath, animal, session, isstim, neuron_n)
%LOGISI Calculate ISI histogram
%   LOGISI(MAINPATH, ANIMAL, SESSION, ISSTIM, NEURON_N) computes and plots
%   normalized inter spike interval (ISI) histogram with logarithmic x
%   axis.
%
%   Required input arguments:
%       MAINPATH: the acces route of the folder containing the results
%           matrix and the database: folders with animal IDs containing
%           folders with session IDs, containing the preprocessed data
%           (clustered spike times, and extracted tSCs in the \raw folder).
%       ANIMAL: ID of the animal
%       SESSION: ID of the session
%       ISSTIM: logical variable indicating whether there are stimulation
%       periods that need to be exluded from the data.
%   Optional input arguments:
%       NEURON_N: if only one neuron should be examined, the number of the
%       neuron

%   Bálint Király
%   Institute of Experimental Medicine, Budapest, Hungary
%   kiraly.balint@koki.hu
%   19-Dec-2021

% Define directories
figfold = 'Spectra';
base = fullfile(mainPath,animal,session);
filename = [animal,session];
mkdir(fullfile(base,figfold))
Matrixname = 'Matrix';

% Load results Matrix
if exist(fullfile(mainPath,[Matrixname,'.mat']), 'file')
    Matrix = load(fullfile(mainPath,[Matrixname,'.mat']));
else
    Matrix = struct('Matrix',struct('ID',{})); % create Matrix if not exist
end
Matrix = Matrix.Matrix;

% Load stimulation times
if isstim
    stim = load_session_stim(mainPath, animal, session);
end

% Main analysis for each cell
list = dir(fullfile(base,'TT*.mat')); % list cells in the data folder
if nargin < 5
    neuron_num = length(list);
    neuron_n = 1;
else
    neuron_num = neuron_n;
end

for neuron = neuron_n:neuron_num % neuron loop
    ID = list(neuron).name(find(list(neuron).name == '_')-1:find(list(neuron).name == '.') - 1); %cell ID number
    spikes = cell2mat(struct2cell(load(fullfile(base,['TT',ID,'.mat'])))); % load spike times

    % Omit spikes during stimulation period
    if isstim == 1
        spikes(stim(spikes) == 1) = [];
    end

    % Plot ISI histogram with logarithmic scale
    figure
    ISI = diff(spikes);
    ISI_ = ISI + rand(length(ISI),1) - 0.5; % make the histogram look continous by adding random deviation from the smapling points
    [isihist_log_norm,edges] = histcounts(ISI_,logspace(0,4,100),'Normalization', 'probability');
    histogram(ISI_,edges,'Normalization', 'probability')
    set(gca,'xscale','log')
    ylabel('Probability')
    xlabel('ISI (ms)')
    saveas(gcf, fullfile(base, figfold, [filename,'_', ID,'logISI.png']));

    % Find the current cell in the Matrix
    M_index = find(strcmp({Matrix.ID}, [filename,'_', ID]) == 1);
    if isempty(M_index)
        M_index = length(Matrix) + 1;
        Matrix(M_index).ID = [filename,'_', ID]; % if not yet in the Matrix add
    end

    % Add/update results to the Matrix
    Matrix(M_index).isihist = isihist_log_norm;

end

% Save results
save(fullfile(mainPath,[Matrixname,'.mat']),'Matrix');
