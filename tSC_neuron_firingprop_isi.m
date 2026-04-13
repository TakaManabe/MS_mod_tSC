function tSC_neuron_firingprop_isi(mainPath, animal, session, ch, isstim, neuron_n)
%TSC_NEURON_FIRINGPROP_ISI Compare ISI histogram during theta cycles expressing and not expressing tSCs
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

% Define directories
figfold = 'Figures';
filename = [animal,session];
[fullpath, base] = resolve_session_fullpath(mainPath, animal, session, ch);
mkdir(fullfile(base,figfold));

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

% Ommit theta cycles during stimulation
if isstim == 1

    if theta_cycles(1,1) < 0 % ommit first theta cycle if the beginning is missing
        theta_cycles(1,:) = [];
        projections(1,:) = [];
    end

    stim = load_session_stim(mainPath, animal, session);
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

    figure
    maximize_figure(gcf);
    tsc_spikes = cell(1,6);
    no_tsc_spikes = cell(1,6);
    ISI_ = cell(1,4);
    ISI_no = cell(1,4);

    % tSC loop
    for tSC = 1:4

        % Find theta cycles with strong tSC
        p_tSC = projections(:,tSC); %distribution of the projections of the given tSC
        trsh = 2 * median(abs(p_tSC - median(p_tSC))) / 0.6745 + median(p_tSC); %define threshold for strong tSCs
        tsc_dom = find(p_tSC > trsh);
        %no_tsc_dom = find(p_tSC < prctile(p_tSC,20));
        SP = sort(p_tSC);
        no_tsc_dom = find(p_tSC < SP(length(tsc_dom)));
        tsc_cycles(tsc_dom) = 1;

        % Calculate mean firing parameters for theta cycles with strong
        % tSC expression
        for index=1:length(tsc_dom)
            tsc_spikes{tSC} = [tsc_spikes{tSC}; spikes(spikes > theta_cycles(tsc_dom(index),1) & spikes < theta_cycles(tsc_dom(index),6))];
        end

        for index=1:length(no_tsc_dom)
            no_tsc_spikes{tSC} = [no_tsc_spikes{tSC}; spikes(spikes > theta_cycles(no_tsc_dom(index),1) & spikes < theta_cycles(no_tsc_dom(index),6))];
        end

        subplot(4,tSC_num + 1,tSC)
        ISI_{tSC}= diff(tsc_spikes{tSC});
        ISI_no{tSC}= diff(no_tsc_spikes{tSC});
        histogram(ISI_{tSC},0:5:70,'FaceColor',Colors(tSC,:))
        hold on
        histogram(ISI_no{tSC},0:5:70,'FaceColor','k')
        set(gca, 'YScale', 'log')
        xlim([0,70])
        setmyplot_balazs
        line([1/mainfreqs(tSC)*1000,1/mainfreqs(tSC)*1000],[10,300])

    end
    % Calculate mean firing parameters for theta cycles without strong
    % tSC expression
    tSC = 6;
    tsc_dom_no = find(tsc_cycles==0);
    for index=1:length(tsc_dom_no)
        tsc_spikes{tSC} = [tsc_spikes{tSC}; spikes(spikes > theta_cycles(tsc_dom_no(index),1) & spikes < theta_cycles(tsc_dom_no(index),5))];
    end

    ISI_{tSC}= diff(tsc_spikes{tSC});

    % Plot theta-phase histogram for cycles without strong tSC
    subplot(4,tSC_num + 1,tSC_num + 1);

    % Plot firing rate bar plot
    subplot(4,tSC_num + 1,(tSC_num + 1));
    histogram(ISI_{6},0:1:70,'Normalization', 'probability')
    xlim([0,70])

    % Save figure
    saveas(gcf, fullfile(base, figfold, [filename,'_', ID,'firingprop_burst_abs.pdf']));

end








