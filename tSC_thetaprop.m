function tSC_thetaprop(mainPath, animal, session, ch, isstim, isrun)
%TSC_THETAPROP Calculates theta properties during tSC presence
%   TSC_THETAPROP(MAINPATH, ANIMAL, SESSION, CH, ISSTIM, ISRUN) is detecting
%   theta cycles expressing different tSCs and calcultes the properties of
%   these cycles including the number of cycles, the theta frequency and
%   the speed of the animal.
%
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
%       ISRUN: logical variable indicating wheter there is movement data
%           for the freely behaving animals.

%   Bálint Király
%   Institute of Experimental Medicine, Budapest, Hungary
%   kiraly.balint@koki.hu
%   28-Oct-2021


% Define directories
base = fullfile(mainPath,animal,session);
folder = 'raw';
filename = [animal,session];
fullpath = fullfile(base,folder,[filename,'_1']);

% Load extracted theta cycles
theta_cycles = cell2mat(struct2cell(load([fullpath,'.theta.cycles.',ch,'.mat'])));
theta_cycles = theta_cycles + 1; % conversion to Matlab indexing
theta_amp = cell2mat(struct2cell(load([fullpath,'.theta.cycleamp.',ch,'.mat'])));

% Load tSCs and find peak frequencies
ICA = load([fullpath,'.tSCs.ica.',ch,'.mat']);
projections = ICA.projections;
freqs_ = ICA.freqs;
tSCs = ICA.tSCs;
ChLabels = ICA.ChLabels;
[~,mainfreqs] = max(tSCs((length(ChLabels) - 1) * length(freqs_) + 1:((length(ChLabels) - 1) * length(freqs_) + length(freqs_)),:));
[~,ind] = sort(mainfreqs,'ascend');
tSCs(:,:)=tSCs(:,ind);
projections(:,:) = projections(:,ind);
mainfreqs = mainfreqs(ind) + double(freqs_(1)) - 1;
tSC_num = length(mainfreqs);

% Ommit theta cycles during stimulation
if isstim == 1
    if theta_cycles(1,1) < 0 % ommit first theta cycle if the beginning is missing
        theta_cycles(1,:) = [];
        projections(1,:) = [];
    end
    stim = cell2mat(struct2cell(load(fullfile(mainPath,'STIMULATIONS',[filename,'.mat']),'stim')));
    s_inx = find(stim(theta_cycles(:,2)) == 1);
    theta_cycles(s_inx,:) = [];
    projections(s_inx,:) = [];
    theta_amp(s_inx) = [];
end

% Load speed data
if isrun
    load(fullfile(base,'Position.mat'))
    tt = POS.t;
    vv = POS.speed;
    t = [tt{1} / 20 ; tt{2} / 20 + L1];
    v = [vv{1} ; vv{2}];
end

% Preallocate memory
all_cycle_speed = NaN(1,length(theta_cycles));
tsc_cycles = zeros(length(theta_cycles),tSC_num);
numberofcycles = zeros(1,6);
avg_cycle_length = zeros(1,6);
avg_cycle_speed = zeros(1,6);
avg_cycle_amp = zeros(1,6);
% Calculate median speed during each theta cycle
for cyc_inx = 1:length(theta_cycles)
    % detect the time borders of the cycle
    try
        t1 = theta_cycles(cyc_inx,1);
        [~,i1] = min(abs(t - double(t1)));
        t2 = theta_cycles(cyc_inx,5);
        [~,i2] = min(abs(t - double(t2)));
        all_cycle_speed(cyc_inx) = median(v(i1 + 1:i2 + 1));
    catch
        all_cycle_speed(cyc_inx) = NaN; % skip if speed data is missing
    end
end

for tSC = 1:1:tSC_num + 1

    if tSC <= tSC_num
        % Find theta cycles with strong tSC
        p_tSC = projections(:,tSC); % distribution of the projections of the given tSC
        trsh = 2 * median(abs(p_tSC - median(p_tSC))) / 0.6745 + median(p_tSC); % define threshold for strong tSCs
        tsc_dom = find(p_tSC > trsh);
        tsc_cycles(tsc_dom,tSC) = 1;
    else % no strong tSC
        tsc_dom = find(sum(tsc_cycles,2) == 0);
        tsc_cycles(tsc_dom,tSC) = 1;
    end

    numberofcycles(tSC) = length(tsc_dom);
    avg_cycle_length(tSC) = mean(theta_cycles(tsc_dom,5) - theta_cycles(tsc_dom,1));
    avg_cycle_amp(tSC) = mean(theta_amp(tsc_dom));
    avg_cycle_speed(tSC) = nanmean(all_cycle_speed(tsc_dom));
end

% Load results Matrix
if exist(fullfile(mainPath,'ses_Matrix.mat'), 'file')
    ses_Matrix = load(fullfile(mainPath,'ses_Matrix.mat'));
    ses_Matrix = ses_Matrix.ses_Matrix;
else
    ses_Matrix = []; % create Matrix if not exist
    ses_Matrix(1).ID = filename;
end

% Find the current session
M_index = find(strcmp({ses_Matrix.ID}, filename) == 1);
if isempty(M_index)
    M_index = length(ses_Matrix) + 1;
    ses_Matrix(M_index).ID = filename;
end

% Save results
ses_Matrix(M_index).mainfreqs = mainfreqs;
ses_Matrix(M_index).tSCs = tSCs((length(ChLabels) - 1) * length(freqs_) + 1:((length(ChLabels) - 1) * length(freqs_) + length(freqs_)),:);
ses_Matrix(M_index).numberofcycles = numberofcycles;
ses_Matrix(M_index).numberofallcycles = length(theta_cycles);
ses_Matrix(M_index).tscStrength = projections;
ses_Matrix(M_index).allcyclelength = theta_cycles(:,5) - theta_cycles(:,1);
ses_Matrix(M_index).avg_cycle_length = avg_cycle_length;
ses_Matrix(M_index).avg_allcycle_length = mean(theta_cycles(:,5) - theta_cycles(:,1));
ses_Matrix(M_index).cycles_speed_all = all_cycle_speed;
ses_Matrix(M_index).avg_cycles_speed = avg_cycle_speed;
ses_Matrix(M_index).avg_allcycle_speed = nanmean(all_cycle_speed);
ses_Matrix(M_index).cycle_amplitudo_all = theta_amp;
ses_Matrix(M_index).avg_cycle_amplitudo = avg_cycle_amp;
ses_Matrix(M_index).avg_allcycle_amplitudo = mean(theta_amp);

save(fullfile(mainPath,'ses_Matrix.mat'),'ses_Matrix');
