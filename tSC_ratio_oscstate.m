function tSC_ratio_oscstate(mainPath, animal, session, ch, isstim, exp)
%TSC_RATIO_OSCSTATE calculates the ratio of different theta cycles during
%different oscillation states.
%   TSC_RATIO_OSCSTATE(MAINPATH, ANIMAL, SESSION, CH, ISSTIM, EXP) detects
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
%   See also THETA_DETECTION

%   Bálint Király
%   Institute of Experimental Medicine, Budapest, Hungary
%   kiraly.balint@koki.hu
%   28-Oct-2021


% Load preset theta state detection settings corresponding to the experiment
global Samplingrate; % in Hz
if strcmp(exp,'awake_mouse')
    thetrange = [4,12];
    deltarange = [0.5,4];
    ratio = 2;
    win = 20;
    win_s = 3;
elseif strcmp(exp,'anesthetized_mouse')
    thetrange = [2,8];
    deltarange = [0.5,2];
    ratio = 1;
    win = 30;
    win_s = 5;
elseif strcmp(exp,'anesthetized_rat')
    thetrange = [3,8];
    deltarange = [0.5,3];
    ratio = 1;
    win = 30;
    win_s = 5;
end

% Define directories
base = fullfile(mainPath,animal,session);
folder = 'raw';
filename = [animal,session];
fullpath = fullfile(base,folder,[filename,'_1']);

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
mainfreqs = mainfreqs(ind) + double(freqs_(1)) - 1;
tSC_num = length(mainfreqs);

% Ommit theta cycles during stimulation
if isstim == 1
    stim = cell2mat(struct2cell(load(fullfile(mainPath,'STIMULATIONS',[filename,'.mat']),'stim')));
    s_inx = find(stim(theta_cycles(:,2)) == 1);
    theta_cycles(s_inx,:) = [];
    projections(s_inx,:) = [];
end

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

% Detect theta states
[theta_long, ~, ~] = theta_detection(eeg', thetrange, deltarange, Samplingrate,win_s, ratio, win);
[theta_short, ~, ~] = theta_detection(eeg', thetrange, deltarange, Samplingrate,win_s, ratio, win_s);
[theta_0_long, ~, ~] = theta_detection(eeg', thetrange, deltarange, Samplingrate,win_s, ratio * 0.75, win);

% Define the ratio theta cycles expressing a given tSC
ratios_longtheta = ratio_calc(theta_long,theta_cycles,tsc_cycles);
ratios_shorttheta = ratio_calc(theta_short>theta_long,theta_cycles,tsc_cycles);
ratios_transition = ratio_calc(theta_long<theta_0_long,theta_cycles,tsc_cycles);

% Load results Matrix
if exist(fullfile(mainPath,'ses_Matrix.mat'), 'file')
    ses_Matrix = load(fullfile(mainPath,'ses_Matrix.mat'));
    ses_Matrix = ses_Matrix.ses_Matrix;
else
    ses_Matrix = struct('ID',{}); % create Matrix if not exist
end

% Find the current session
M_index=find(strcmp({ses_Matrix.ID}, filename) == 1);
if isempty(M_index)
    M_index = length(ses_Matrix) + 1;
    ses_Matrix(M_index).ID=filename;
end

% Save results
ses_Matrix(M_index).shorttheta = ratios_shorttheta;
ses_Matrix(M_index).margin_long = ratios_transition;
ses_Matrix(M_index).longtheta = ratios_longtheta;
save(fullfile(mainPath,'ses_Matrix.mat'),'ses_Matrix');


% -------------------------------------------------------------------------
function ratio = ratio_calc(osc_state,theta_cycles,tsc_cycles)
% Calculate the ratio of theta cycles expressing a given tSC during the
% oscillation state defined by osc_state
ratio = zeros (1,size(tsc_cycles,2) + 1);
for tSC = 1:size(tsc_cycles,2)
    Number_of_tSC_cycles = tsc_cycles((osc_state(theta_cycles(:,2)) == 1),tSC);
    ratio(tSC) = sum(Number_of_tSC_cycles) / length(find(osc_state(theta_cycles(:,2)) == 1));
end
Number_of_all_cycles = sum((osc_state(theta_cycles(:,2)) == 1));
ratio(size(tsc_cycles,2) + 1) = sum(Number_of_all_cycles) / length(theta_cycles);
