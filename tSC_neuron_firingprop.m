function tSC_neuron_firingprop(mainPath, animal, session, ch, isstim, neuron_n)
%TSC_NEURON_FIRINGPROP  Main preprocessing of spikes - tSC connection data.
%   TSC_NEURON_FIRINGPROP(MAINPATH, ANIMAL, SESSION, CH, ISSTIM, NEURON_N)
%   is calculating mean firing properties for single neuron activities
%   recorded simultaneously with LFP oscilattions containing theta nested
%   spectral components (tSCs).
%   Firing properties during theta cycles involve the firing rate,
%   intraburst spike number, intraburst frequency, burst-skip ratio,
%   theta phase coupling strength and the preferred theta phase. Each
%   property is also calculated selectively for theta cycles expressing a
%   given tSC.
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
%       periods that need to be exluded from the data.
%   Optional input arguments:
%       NEURON_N: if only one neuron should be examined, the number of the
%       neuron

%   Bálint Király
%   Institute of Experimental Medicine, Budapest, Hungary
%   kiraly.balint@koki.hu
%   25-Oct-2021


% Settings
global Colors
global Samplingrate
hist_res = 15; % number of bins in the phase histogram
maxburstL = 40; % maximum threshold for intraburst interval (ms)

% Define directories
figfold = 'Figures';
filename = [animal,session];
[fullpath, base] = resolve_session_fullpath(mainPath, animal, session, ch);
mkdir(fullfile(base,figfold));
Matrixname = 'Matrix';

% Load results Matrix
if exist(fullfile(mainPath,[Matrixname,'.mat']), 'file')
    Matrix = load(fullfile(mainPath,[Matrixname,'.mat']));
    Matrix = Matrix.Matrix;
else
    Matrix.ID = []; % create Matrix if not exist
end

% Load extracted emds
theta_cycles = cell2mat(struct2cell(load([fullpath,'.theta.cycles.',ch,'.mat'])));
theta_cycles = theta_cycles + 1; % conversion to Matlab indexing
theta_phase = cell2mat(struct2cell(load([fullpath,'.theta.phase.',ch,'.mat'])));

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

    if theta_cycles(1,1) < 0 % ommit first theta cycle if the beginning is missing
        theta_cycles(1,:) = [];
        projections(1,:) = [];
    end

    stim = load_session_stim(mainPath, animal, session);
    theta_phase(stim == 1) = NaN;
    s_inx = find(stim(theta_cycles(:,2)) == 1);
    theta_cycles(s_inx,:) = [];
    projections(s_inx,:) = [];
end

% Main analysis for each cell
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

    % Find bursts and calculate burst properties for each theta cycle
    [burstrate,burstlength,burstduration,skipinx] = burstprop_per_cycle(spikes,maxburstL,theta_cycles,Samplingrate);

    tsc_cycles = zeros(1,length(theta_cycles));
    phase_hist = NaN(tSC_num + 2,hist_res);
    mfrate = NaN(1,tSC_num + 2);
    stdfrate = NaN(1,tSC_num + 2);
    mean_burstlength = NaN(1,tSC_num + 2);
    mean_burstduration = NaN(1,tSC_num + 2);
    mean_burstrate = NaN(1,tSC_num + 2);
    skipratio = NaN(1,tSC_num + 2);
    hang = NaN(1,tSC_num + 2);
    hmvl = NaN(1,tSC_num + 2);
    Z = NaN(1,tSC_num + 2);
    pRayleigh = NaN(1,tSC_num + 2);
    phase_neuron_sum = cell(1,tSC_num + 2);
    figure
    maximize_figure(gcf);

    % tSC loop
    for tSC = 1:tSC_num

        % Find theta cycles with strong tSC
        p_tSC = projections(:,tSC); %distribution of the projections of the given tSC
        trsh = 2 * median(abs(p_tSC - median(p_tSC))) / 0.6745 + median(p_tSC); %define threshold for strong tSCs
        tsc_dom = find(p_tSC > trsh);
        tsc_cycles(tsc_dom) = 1;

        % Calculate mean firing parameters for theta cycles with strong
        % tSC expression
        [phase_hist(tSC,:),mfrate(tSC),stdfrate(tSC),mean_burstlength(tSC),mean_burstduration(tSC),mean_burstrate(tSC),skipratio(tSC),hang(tSC),hmvl(tSC),phase_neuron_sum{tSC},Z(tSC),pRayleigh(tSC)] = firingprop_per_thetatype(tsc_dom,theta_cycles,spikes,Samplingrate,theta_phase,burstlength,burstduration,burstrate,skipinx,hist_res);

        % Theta phase histrogramplot
        subplot(4,tSC_num + 1,tSC);
        phase_hist_plot(phase_hist(tSC,:),hist_res,tSC);
        title(sprintf('tsc%0.0f (%0.0f Hz) dominant cycles \n Mean firing rate: %0.2f Hz\n Ratio of skipped cycles: %0.2f \n Mean burstlength: %0.2f spikes\n Mean burstrate: %0.2f Hz',tSC,mainfreqs(tSC),mfrate(tSC),skipratio(tSC),mean_burstlength(tSC),mean_burstrate(tSC)));
    end

    % Calculate mean firing parameters for theta cycles without strong
    % tSC expression
    tsc_dom_no = find(tsc_cycles==0);
    [phase_hist(tSC+1,:),mfrate(tSC+1),stdfrate(tSC+1),mean_burstlength(tSC+1),mean_burstduration(tSC+1),mean_burstrate(tSC+1),skipratio(tSC+1),hang(tSC+1),hmvl(tSC+1),phase_neuron_sum{tSC+1},Z(tSC+1),pRayleigh(tSC+1)] = firingprop_per_thetatype(tsc_dom_no,theta_cycles,spikes,Samplingrate,theta_phase,burstlength,burstduration,burstrate,skipinx,hist_res);

    % Plot theta-phase histogram for cycles without strong tSC
    subplot(4,tSC_num + 1,tSC_num + 1);
    phase_hist_plot(phase_hist(tSC+1,:),hist_res,tSC+1);
    title(sprintf('Cycles without strong tsc \n Mean firing rate: %0.2f Hz\n Ratio of skipped cycles: %0.2f \n Mean burstlength: %0.2f spikes \n Mean burstrate: %0.2f Hz',mfrate(tSC+1), skipratio(tSC+1),mean_burstlength(tSC+1),mean_burstrate(tSC+1)));

    % Plot firing rate bar plot
    subplot(4,tSC_num + 1,(tSC_num + 1) * 2);
    bar(mfrate(1:end - 1));
    hold on
    errorbar(1:tSC_num + 1, mfrate(1:end - 1), stdfrate(1:end - 1),'LineStyle','none');
    xlim([0,tSC_num + 2]);
    ylabel('firing rate (Hz)');
    set(gca,'xtick',1:tSC_num + 1);
    set(gca,'xticklabel',({'tSC1','tSC2','tSC3','tSC4','tSC5','no tSC'}));
    set(gca,'XTickLabelRotation',45);

    % Save figure
    saveas(gcf, fullfile(base, figfold, [filename,'_', ID,'firingprop.png']));

    % Calculate mean firing parameters for all theta cycles
    [phase_hist(tSC+2,:),mfrate(tSC+2),stdfrate(tSC+2),mean_burstlength(tSC+2),mean_burstduration(tSC+2),mean_burstrate(tSC+2),skipratio(tSC+2),hang(tSC+2),hmvl(tSC+2),phase_neuron_sum{tSC+2},Z(tSC+2),pRayleigh(tSC+2)] = firingprop_per_thetatype(1:length(tsc_cycles),theta_cycles,spikes,Samplingrate,theta_phase,burstlength,burstduration,burstrate,skipinx,hist_res);

    % Find the current cell
    M_index = find(strcmp({Matrix.ID}, [filename,'_', ID]) == 1);
    if isempty(M_index)
        M_index = length(Matrix) + 1;
        Matrix(M_index).ID = [filename,'_', ID]; % if not yet in the Matrix add
    end

    % Add/update results to the Matrix
    Matrix(M_index).frate = mfrate;
    Matrix(M_index).stdfrate = stdfrate;
    Matrix(M_index).burstlength = mean_burstlength;
    Matrix(M_index).burstduration = mean_burstduration;
    Matrix(M_index).burstrate = mean_burstrate;
    Matrix(M_index).skipratio = skipratio;
    Matrix(M_index).hang = hang;
    Matrix(M_index).hmvl = hmvl;
    Matrix(M_index).Z = Z;
    Matrix(M_index).pRayleigh = pRayleigh;
    Matrix(M_index).phase_hist = phase_hist;
    Matrix(M_index).phase_neuron = phase_neuron_sum;

end

% Save results
save(fullfile(mainPath,[Matrixname,'.mat']),'Matrix');


function [burstrate,burstlength,burstduration,skipinx] = burstprop_per_cycle(neuron,maxburstL,cycles,samplingrate)
% Calculate burst properties of the neuron for each theta cycle

% Find burst starts
[burstWind] = burst_detector(neuron,maxburstL,neuron(end));
burst_start = find(diff([0 burstWind]) == 1);
% preallocate space
skipinx = NaN(length(cycles),1);
burstlength = NaN(length(cycles),1);
burstduration = NaN(length(cycles),1);
burstrate = NaN(length(cycles),1);

for i = 1:length(cycles)
    if sum(burst_start > cycles(i,1) & burst_start < cycles(i,end - 1)) == 0 %no burst initiated in the cycle
        skipinx(i) = 1;
    else % if there are bursts, calculate properties, in case of multiple bursts in a theta cycle take the average
        index = find(burst_start > cycles(i,1) & burst_start < cycles(i,end - 1));
        L = NaN(1,length(index));
        burstlength_temp = NaN(1,length(index));
        for z = 1:length(index)
            L(z) = find(burstWind(burst_start(index(z)):end) == 0,1);
            burstlength_temp(z) = sum((ismember(neuron,(burst_start(index(z)):burst_start(index(z)) + L(z)))));
        end
        burstlength(i) = mean(burstlength_temp);
        burstduration(i) = mean(L);
        burstrate(i) = (burstlength(i) - 1) / burstduration(i) * samplingrate;
    end
end
skipinx=find(skipinx==1);


function [phase_hist,mfrate,stdfrate,mean_burstlength,mean_burstduration,mean_burstrate,skipratio,hang,hmvl,phase_neuron_sum,Z,pRayleigh] ...
    = firingprop_per_thetatype(tsc_dom,cycles,neuron,samplingrate,phase,burstlength,burstduration,burstrate,skipinx,hist_res)
% Calculate mean firing properties of the neuron across the theta cycles defined in tsc_dom

% Preallocate space
phase_theta = NaN(1,length(phase));
frate=NaN(1,length(tsc_dom));

% Calculate firing rate and spike phase values from each cycle
for i = 1:length(tsc_dom)
    cyc(1) = double(cycles(tsc_dom(i),1));
    cyc(2) = double(cycles(tsc_dom(i),5));
    frate(i) = length(find(neuron > cyc(1) & neuron < cyc(end))) / (cyc(end) - cyc(1)) * samplingrate;
    phase_theta(cyc(1):cyc(end)) = phase(cyc(1):cyc(end));
end

% Calculate mean properties
phase_neuron = phase_theta(neuron);
phase_neuron_sum = phase_neuron(~isnan(phase_neuron));
ftm = sum(exp(1).^(1i * phase_neuron_sum)) / length(phase_neuron_sum);    % first trigonometric moment
hang = angle(ftm);   % mean angle
hmvl = abs(ftm);     % mean resultant length
try
    [Z,pRayleigh,~,~] = b_rao(phase_neuron_sum); % Rayleigh's test
catch
    Z = NaN;
    pRayleigh = NaN;
end
phase_hist = hist(phase_neuron_sum,hist_res) / length(tsc_dom);
mfrate = mean(frate);
stdfrate = std(frate);
mean_burstlength = nanmean(burstlength(tsc_dom));
mean_burstduration = nanmean(burstduration(tsc_dom));
mean_burstrate = nanmean(burstrate(tsc_dom));
skipratio = sum(ismember(tsc_dom,skipinx)) / length(tsc_dom);

function  phase_hist_plot(phase_hist,hist_res,tSC)
global Colors
bar(phase_hist,'FaceColor',Colors(tSC,:));
xlim([0,hist_res + 1]);
ylim([0,1]);
set(gca,'xtick',1:7:15)
set(gca,'xticklabel',({'0','pi','2pi'}))
xlabel('hippocampal thate phase')
hold on
tt = (0:0.1:2*pi) / (2 * pi) * (hist_res + 1);
plot(tt,cos(tt * (2 * pi) / (hist_res + 1)) * 0.05 + 0.3,'r')
ylabel('phase coupling')

