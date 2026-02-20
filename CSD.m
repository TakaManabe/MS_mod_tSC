function csd = CSD(mainPath, animal, exp, session, chlist)
%CSD Current source densitiy analysis
%   CSD(MAINPATH,ANIMAL,EXP,SESSION,CHLIST) performs current source
%   densitiy analysis of LFP data recorded with several equidistant silicon
%   probes sites from a laminar structure. CSD signal for each channel
%   (except the borders) is saved for further analysis.
%
%   Required input arguments:
%       MAINPATH: the acces route of the folder containing the results
%           matrix and the database: folders with animal IDS containing
%           folders with experiment IDS, containing the preprocessed data
%           (clustered spike times, and extracted tSCs in the \raw folder).
%       ANIMAL: ID of the animal
%       EXP: ID of the experiment
%       SESSION: number of the session
%       CHLIST: ordered list of channel numbers referring to distinc data
%       files

%   Bálint Király
%   Institute of Experimental Medicine, Budapest, Hungary
%   kiraly.balint@koki.hu
%   30-Oct-2021

% Define directories
base = fullfile(mainPath,animal,exp);
folder = 'raw';
filename = [animal,exp];
fullpath = fullfile(base,folder,[filename,'_',session]);
csd_dir = fullfile(base,folder,'csd');
mkdir(csd_dir)

% Calculate and save CSD based on the LFP of the prevoius, the curren and
% the following channel.
for i=2:length(chlist)-1 %loop through channels except the borders
        [eegpre]=cell2mat(struct2cell(load([fullpath,'.eeg.', int2str(chlist(i - 1)),'.mat'])));
        [eeg]=cell2mat(struct2cell(load([fullpath,'.eeg.', int2str(chlist(i)),'.mat'])));
        [eegpost]=cell2mat(struct2cell(load([fullpath,'.eeg.', int2str(chlist(i + 1)),'.mat'])));
        csd = -(eegpre - 2 * eeg + eegpost);
        savepath_csd = fullfile(csd_dir,[filename,'_',session,'.eeg.', int2str(chlist(i)),'.mat']);
        save(savepath_csd,'csd','-v7')
end
