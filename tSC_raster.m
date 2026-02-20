function tSC_raster (mainPath, animal, session, ch)
%TSC_RASTER Plot raw LFP with spike rasters for single units
%   TSC_RASTER(MAINPATH, ANIMAL, SESSION, CH)
%
%   Required input arguments:
%       MAINPATH: the acces route of the folder containing the database:
%       folders with animal IDs containing folders with session IDs,
%       containing the preprocessed data (clustered spike times, and
%       downsampled LFPs in the \raw folder).
%       ANIMAL: ID of the animal
%       SESSION: ID of the session
%       CH: number of the LFP channel
%
%   See also PLOT_RASTER_LINES_FAST
%   Bálint Király
%   Institute of Experimental Medicine, Budapest, Hungary
%   kiraly.balint@koki.hu
%   20-Dec-2021

% Define directories
base = fullfile(mainPath,animal,session);
folder = 'raw';
filename = [animal,session];
fullpath = fullfile(base,folder,[filename,'_1']);
% load hippocampal LFP
eeg = cell2mat(struct2cell(load([fullpath,'.eeg.', ch ,'.mat'])));

figure
ax1 = subplot(2,1,1);
plot(eeg)
ax2 = subplot(2,1,2);
hold on
j = 1;
% Main analysis for each cell
for shank = 1:1:4 % shank loop

    list = dir(fullfile(base,['TT',int2str(shank),'*.mat'])); % list cells in the data folder

    for neuron = 1:length(list) % neuron loop
        ID = list(neuron).name(find(list(neuron).name == '_')+1:find(list(neuron).name == '.') - 1); %cell ID number
        spikes = cell2mat(struct2cell(load(fullfile(base,['TT',int2str(shank),'_',ID,'.mat'])))); % load spike times

        plot_raster_lines_fast(spikes,[j,j+1]);
        j = j + 1;
    end
end

linkaxes([ax1,ax2],'x')
