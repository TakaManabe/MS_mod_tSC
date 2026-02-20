function tsc_STA(mainPath, animal, session, ch, isstim, neuron_n)
%TSC_STA Spike triggered average analysis during theta cycles with tSCs
%   TSC_STA(MAINPATH,ANIMAL,SESSION,CH,ISSTIM,NEURON_N) performs spike
%   triggered average analysis on LFP and on the Wavelet power and phase of
%   the LFP. The analysis is repeated for spikes during theta cycles
%   expressing different tSCs.
%
%   Required input arguments:
%       MAINPATH: the acces route of the folder containing the results
%           matrix and the database: folders with animal IDs containing
%           folders with session IDs, containing the preprocessed data
%           (clustered spike times, and extracted tSCs in the \raw folder).
%       ANIMAL: ID of the animal
%       SESSION: ID of the session
%       CH: number of the channel used for tSC extraction
%       ISSTIM: logical variable indicating whether there are stimulation
%           periods that need to be exluded from the data.
%   Optional input arguments:
%       NEURON_N: if only one neuron should be examined, the number of the
%       neuron
%
%   See also ETA

%   Bálint Király
%   Institute of Experimental Medicine, Budapest, Hungary
%   kiraly.balint@koki.hu
%   29-Oct-2021


% Settings
global Samplingrate; % sampling rate (Hz)
time_window = 100; % +/- time window size around spikes (ms)

% Define directories
figfold = 'spiketriggeredfigs';
base = fullfile(mainPath,animal,session);
folder = 'raw';
filename = [animal,session];
fullpath = fullfile(base,folder,[filename,'_1']);
mkdir(fullfile(base,figfold))

% Load extracted emds
theta_cycles = cell2mat(struct2cell(load([fullpath,'.theta.cycles.',ch,'.mat'])));
theta_cycles = theta_cycles + 1; % conversion to Matlab indexing
eeg = cell2mat(struct2cell(load([fullpath,'.eeg.', ch ,'.mat'])));

% Load tSCs and find peak frequencies
ICA = load([fullpath,'.tSCs.ica.',ch,'.mat']);
projections = ICA.projections;
freqs_ = ICA.freqs;
tSCs = ICA.tSCs;
ChLabels = ICA.ChLabels;
[~,mainfreqs] = max(tSCs((length(ChLabels) - 1) * length(freqs_) + 1:((length(ChLabels) - 1) * length(freqs_) + length(freqs_)),:));
[~,ind] = sort(mainfreqs,'ascend');
projections(:,:) = projections(:,ind);
tSC_num = length(mainfreqs);
stim = cell2mat(struct2cell(load(fullfile(mainPath,'STIMULATIONS',[filename,'.mat']),'stim')));

% STA analysis for each cell during theta cycles expressing the different
% tSCs
list = dir(fullfile(base,'TT*.mat')); % list cells in the data folder
if nargin < 6
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

    figure
    for tSC = 1:1:tSC_num + 1 % Loop through tSCs and 'All spikes' at last
        % Find spikes during cycles strongly expressing a tSC
        if tSC <= tSC_num
            % Find theta cycles with strong tSC
            p_tSC = projections(:,tSC); % distribution of the projections of the given tSC
            trsh = 2 * median(abs(p_tSC - median(p_tSC))) / 0.6745 + median(p_tSC); % define threshold for strong tSCs
            tsc_dom = find(p_tSC > trsh);
            % Omit other spikes
            filtered_spikes = cell(1,1);
            for spike_inx = 1:length(spikes)
                cycinx = find(spikes(spike_inx) > theta_cycles(tsc_dom,2),1,'last');
                if spikes(spike_inx) < theta_cycles(tsc_dom(cycinx),6)
                    filtered_spikes{1} = [filtered_spikes{1} spikes(spike_inx)];
                end
            end
            filtered_spikes = cell2mat(filtered_spikes);
        else % all spikes
            filtered_spikes = spikes;
        end

        ETA(filtered_spikes,eeg,Samplingrate,time_window,tSC_num,tSC);
        subplot(3,tSC_num + 1,tSC)
        if tSC == 6
            title('All spikes')
        else
            title(['tSC',int2str(tSC)]);
        end
    end

    % Save figure
    maximize_figure(gcf)
    savefig(fullfile(base, figfold, ['tSC', filename,'_', ID]))
    close all
end
