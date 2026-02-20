function stimfeedback_tSC(mainPath, animal, session, ch)
%TSC_RATIO_OSCSTATE calculates the ratio of different theta cycles during
%stimulation and control periods.
%   stimfeedback_tSC(MAINPATH, ANIMAL, SESSION, CH, ISSTIM, EXP) detects
%   oscillation states in the hippocampal LFP (long and short theta epocs,
%   theta-delta transitions) and compares the ratio of theta cycles
%   strongly expressing different tSCs during each oscillation state.

%   Required input arguments:
%       MAINPATH: the acces route of the folder containing the results
%           matrix and the database: folders with animal IDs containing
%           folders with session IDs, containing the raw data and
%           extracted tSCs in the \raw folder).
%       ANIMAL: ID of the animal
%       SESSION: ID of the session
%       CH: number of the channel used for tSC extraction
%       ISSTIM: logical variable indicating whether there are stimulation
%           periods that need to be exluded from the data.
%       EXP: name of the experiment used for choosing between
%           preset settings for oscillation state detection defined by
%           Kocsis et al, 2020 . Possible inputs: 'awake_mouse',
%           'anesthetized_mouse', 'anesthetized_rat'.
%

%   Bálint Király
%   Institute of Experimental Medicine, Budapest, Hungary
%   kiraly.balint@koki.hu
%   28-Oct-2021


% Load preset theta state detection settings corresponding to the experiment
global Samplingrate; % in Hz

% Define directories
base = fullfile(mainPath,animal,session);
folder = 'raw';
filename = [animal,session];
fullpath = fullfile(base,folder,[filename,'_1']);

% Load extracted emds
theta_cycles = cell2mat(struct2cell(load([fullpath,'.theta.cycles.',ch,'.mat'])));
theta_cycles = theta_cycles + 1; % conversion to Matlab indexing

% Load tSCs and find peak frequencies
ICA = load([fullpath,'.tSCs.ica.',ch,'.mat']);
projections = ICA.projections;
freqs_ = ICA.freqs;
tSCs = ICA.tSCs;
ChLabels = ICA.ChLabels;
[~,mainfreqs] = max(tSCs((length(ChLabels) - 1) * length(freqs_) + 1:((length(ChLabels) - 1) * length(freqs_) + length(freqs_)),:));
[~,ind] = sort(mainfreqs,'ascend');
projections(:,:) = projections(:,ind);
mainfreqs = mainfreqs(ind) + double(freqs_(1)) - 1;
tSC_num = length(mainfreqs);

% Preallocate memory
tsc_cycles = zeros(length(theta_cycles),tSC_num);

for tSC = 1:1:tSC_num + 1
    if tSC <= tSC_num
        % Find theta cycles with strong tSC
        p_tSC = projections(:,tSC); % distribution of the projections of the given tSC
        trsh = 2 * median(abs(p_tSC - median(p_tSC))) / 0.6745 + median(p_tSC); % define threshold for strong tSCs
        tsc_cycles(p_tSC > trsh,tSC)=1;
    else % find theta cycles without strong tSC
        tsc_cycles(sum(tsc_cycles,2) == 0,tSC) = 1;
    end
end

stim = cell2mat(struct2cell(load(fullfile(mainPath,'STIMULATIONS',[filename,'.mat']),'stim')));
[temp] = unifying_and_short_killer(stim, Samplingrate, 3);
[stim] = unifying_and_short_killer(stim, Samplingrate, 1.5);
stim(stim==0 & temp==1) = nan;

s_inx = find(stim(theta_cycles(:,2)) == 1);

ns_inx = find(stim(theta_cycles(:,2)) == 0);

inhib_tsc_ratio = mean(tsc_cycles(s_inx,:));
noninhib_tsc_ratio = mean(tsc_cycles(ns_inx,:));


% Load results Matrix
if exist(fullfile(mainPath,'ses_Matrix.mat'), 'file')
    ses_Matrix = load(fullfile(mainPath,'ses_Matrix.mat'));
    ses_Matrix = ses_Matrix.ses_Matrix;
else
    ses_Matrix(1).ID=filename;; % create Matrix if not exist
end

% Find the current session
M_index=find(strcmp({ses_Matrix.ID}, filename) == 1);
if isempty(M_index)
    M_index = length(ses_Matrix) + 1;
    ses_Matrix(M_index).ID=filename;
end

% Save results
ses_Matrix(M_index).inhib_tsc_ratio = inhib_tsc_ratio;
ses_Matrix(M_index).noninhib_tsc_ratio = noninhib_tsc_ratio;
save(fullfile(mainPath,'ses_Matrix.mat'),'ses_Matrix');
