function tmS_STA(mainPath, animal, session, ch, stimepoch, burstlength, time_window, columnsnum, column)
%TMS_STA Triggered average analysis for theta modulated stimulation bursts
%   TMS_STA(MAINPATH,ANIMAL,SESSION,CH,STIMEPOCH,BURSTLENGTH,TIME_WINDOW,COLUMNSNUM,COLUMN)
%   performs stimulus triggered average analysis on LFP and on the Wavelet
%   power and phase of the LFP.
%
%   Required input arguments:
%       MAINPATH: the acces route of the folder containing the database:
%       folders with animal IDs containing folders with session IDs,
%       containing the extracted tSCs in the \raw folder).
%       ANIMAL: ID of the animal
%       SESSION: ID of the session
%       CH: number of the LFP channel
%       STIMEPOCH: which stimulation epoch to examine within the session
%       BURSTLENGTH: number of stimulies in a stimulation burst
%       TIME_WINDOW: % +/- size of the time window around events (ms)
%       COLUMNSNUM: number of columns plotted
%       COLUMN: number of the current column

%   See also ETA and TSC_STA

%   Bálint Király
%   Institute of Experimental Medicine, Budapest, Hungary
%   kiraly.balint@koki.hu
%   23-Dec-2021

global Samplingrate

base = fullfile(mainPath,animal,[animal,'_',session],'raw');
filename = [animal,'_',session];

eeg = cell2mat(struct2cell(load(fullfile(base,[filename,'.eeg.', ch ,'.mat']))));

[~,t]=load_open_ephys_data(fullfile(base,'100_CH1.continuous'));
[~,stimt]=load_open_ephys_data(fullfile(base,'all_channels.events'));
%timestamps=readNPY([mainpath,'timestamps', '.npy']);
%stims=(readNPY([mainpath,'timestamps_stim', '.npy'])-timestamps(1))/20;

stims = round((stimt - t(1)) * Samplingrate);
interburst_threshold = 200;
borders=[1;find(diff(stims) > interburst_threshold) + 1];

ETA(double(stims(borders(stimepoch):burstlength * 2:borders(stimepoch + 1))),eeg,Samplingrate,time_window,columnsnum,column)
