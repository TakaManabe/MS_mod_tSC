function tSC_neuron_coupling(mainPath, animal, session, ch, isstim, isshuffle, neuron_n)
%TSC_NEURON_COUPLING  testing phase coupling to tSCs
%   TSC_NEURON_FIRINGPROP(MAINPATH, ANIMAL, SESSION, CH, ISSTIM, ISSHUFFLE,
%   NEURON_N) performs Rayleigh's test for circular uniformity on
%   temporallyshifted spikes to test phase coupling to tSCs and the
%   temporal relation of the two signal.
%
%   Required input arguments:
%       MAINPATH: the acces route of the folder containing the results
%           matrix and the database: folders with animal IDs containing
%           folders with session IDs, containing the preprocessed data
%           (clustered spike times, and extracted tSCs in the \raw folder).
%       ANIMAL: ID of the animal
%       SSESSION: ID of the session
%       CH: number of the channel used for tSC extraction
%       ISSTIM: logical variable indicating whether there are stimulation
%       periods that need to be exluded from the data.
%       ISSHUFFLE: logical variable indicating whether spikes should be
%       shuffled between theta cycles for the controll analysis
%   Optional input arguments:
%       NEURON_N: if only one neuron should be examined, the number of the
%       neuron

%   Bálint Király
%   Institute of Experimental Medicine, Budapest, Hungary
%   kiraly.balint@koki.hu
%   11-Nov-2021


% Settings
global Samplingrate % sampling rate (Hz)
global Colors % tSC colors
kernels = [5,4,3,2,1]; % kernel size for psth smoothing for each tSC from tSC1 to tSC5
time_window = 100; % +/- time window size for rasters around tSCs (ms)

% Define directories
base = fullfile(mainPath,animal,session);
figfold = 'Figures';
mkdir(fullfile(base,figfold));
folder = 'raw';
filename = [animal,session];
fullpath = fullfile(base,folder,[filename,'_1']);
if isshuffle
    Matrixname = 'Matrix_shuffle';
else
    Matrixname = 'Matrix';
end

% Load results Matrix
if exist(fullfile(mainPath,[Matrixname,'.mat']), 'file')
    Matrix = load(fullfile(mainPath,[Matrixname,'.mat']));
    Matrix = Matrix.Matrix;
else
    Matrix = []; % create Matrix if not exist
end


% Load extracted emds
emd = cell2mat(struct2cell(load([fullpath,'.emd.',ch,'.mat']))); % emds
emdif = cell2mat(struct2cell(load([fullpath,'.emd.if.',ch,'.mat']))); % instantaneous frequencies
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

    stim = cell2mat(struct2cell(load(fullfile(mainPath,'STIMULATIONS',[filename,'.mat']),'stim')));
    theta_phase(stim == 1) = NaN;
    s_inx = find(stim(theta_cycles(:,2)) == 1);
    theta_cycles(s_inx,:) = [];
    projections(s_inx,:) = [];
end

% Main analysis for each cell
list = dir(fullfile(base,'TT*.mat')); % list cells in the data folder
if nargin < 7
    neuron_num = length(list);
    neuron_n = 1;
else
    neuron_num = neuron_n;
end

for neuron = neuron_n:neuron_num % neuron loop
    ID = list(neuron).name(find(list(neuron).name == '_')-1:find(list(neuron).name == '.') - 1); %cell ID number
    spikes = cell2mat(struct2cell(load(fullfile(base,['TT',ID,'.mat'])))); % load spike times

    % Preallocate space and initialize figure
    tsc_cycles = zeros(1,length(theta_cycles));
    phaserel2tsc = cell(1,tSC_num);
    phaserel2tsc_Z_t = cell(1,tSC_num);
    Zshift_tsc = NaN(tSC_num,time_window * 2 + 1);
    pRayleigh_Zshift_tsc = NaN(tSC_num,time_window * 2 + 1);
    figure
    maximize_figure(gcf);

    % tSC loop
    for tSC = 1:tSC_num

        % Find theta cycles with strong tSC
        p_tSC = projections(:,tSC); %distribution of the projections of the given tSC
        trsh = 2 * median(abs(p_tSC - median(p_tSC))) / 0.6745 + median(p_tSC); %define threshold for strong tSCs
        tsc_dom = find(p_tSC > trsh);
        tsc_cycles(tsc_dom) = 1;

        % Preallocate space
        raszt = zeros(time_window * 2,length(tsc_dom));
        phaserel2tsc_zshift = cell(1,time_window * 2 + 1);
        phaserel2tsc_2cyc = cell(1);
        sumemd = zeros(length(tsc_dom),time_window * 2 + 1);

        for cycinx = 1:length(tsc_dom)

            % Find largest tSC trough
            emd_c = median((emdif(:,theta_cycles(tsc_dom(cycinx),2):theta_cycles(tsc_dom(cycinx),end))),2);
            [~,inx] = min(abs(emd_c(2:end) - mainfreqs(tSC)));
            inx = inx + 1;
            [~,inx2] = min(emd(inx,theta_cycles(tsc_dom(cycinx),2):theta_cycles(tsc_dom(cycinx),end)));
            trig = theta_cycles(tsc_dom(cycinx),2) + inx2 - 1;
            tau_extendedwindow = time_window * 2.5;

            % Ommit cycles withouth enough data before or after the tSC trough
            if trig < tau_extendedwindow || length(theta_phase) - trig < tau_extendedwindow
                continue
            end

            % Shuffle spikes between theta cycles
            if isshuffle
                shuffler = randi(length(theta_cycles),1);
                shift = double(theta_cycles(shuffler,2) - theta_cycles(tsc_dom(cycinx),2));
                spikes_s = spikes - shift;
            else
                spikes_s = spikes;
            end

            raszt(:,cycinx) = histcounts(spikes_s,trig - time_window:trig + time_window);
            % Shift spike with different temporal lags
            try
                % Define tSC phase
                wav = emd(inx,trig - tau_extendedwindow:trig + tau_extendedwindow);
                Ht = hilbert(wav);
                z_phase_wholecycle = angle(Ht);
                valleys = find(diff(z_phase_wholecycle) < -5);
                [~,I] = min(abs(valleys - length(wav) / 2));
                z_phase = z_phase_wholecycle(valleys(I - 1):valleys(I + 1));
                a_phase = [((z_phase_wholecycle(valleys(I - 2)+ 1:valleys(I - 1))) + pi - 4 * pi),((z_phase_wholecycle(valleys(I - 1)+ 1:valleys(I))) + pi - 2 * pi),((z_phase_wholecycle(valleys(I) + 1:valleys(I + 1))) + pi),((z_phase_wholecycle(valleys(I + 1) + 1:valleys(I + 2))) + pi + 2 * pi)];

                % Get phase values at spike times
                for tau = -time_window:1:time_window
                    raszt_zshift = histcounts(spikes_s,trig + valleys(I - 1) - tau_extendedwindow - 1 + tau:trig + valleys(I + 1) - tau_extendedwindow - 1 + tau);
                    phaserel2tsc_zshift{tau + time_window + 1}=[phaserel2tsc_zshift{tau + time_window + 1},z_phase(raszt_zshift > 0)];
                end
                raszt_zshift = histcounts(spikes_s,trig + valleys(I - 2) - tau_extendedwindow - 1:trig + valleys(I + 2) - tau_extendedwindow - 1);
                phaserel2tsc_2cyc{1} = [phaserel2tsc_2cyc{1},a_phase(raszt_zshift == 1)];
            catch
                warning('Shifting spikes failed in a cycle')
            end
            % Average tSC
            sumemd(cycinx,:) = emd(inx,trig - time_window:trig + time_window);

        end

        % Save phase values for each tSC
        phaserel2tsc{tSC} = phaserel2tsc_2cyc{1};
        phaserel2tsc_Z_t{tSC} = phaserel2tsc_zshift;

        % Detect the tSC with the minimal spike-tSC_phase data
        if tSC == 1
            min_spikes = max(cellfun('size',phaserel2tsc_zshift,2));
        end
        if max(cellfun('size',phaserel2tsc_zshift,2)) < min_spikes
            min_spikes = max(cellfun('size',phaserel2tsc_zshift,2));
        end

        % Calculate Rayleigh's Z and p for different temporal shifts
        for tau = -time_window:1:time_window
            try
                [Zshift_tsc(tSC,tau + time_window + 1),pRayleigh_Zshift_tsc(tSC,tau+time_window + 1),~,~] = b_rao(phaserel2tsc_zshift{tau + time_window + 1});
            catch
                Zshift_tsc(tSC,tau + time_window + 1) = NaN;
                pRayleigh_Zshift_tsc(tSC,tau + time_window + 1) = NaN;
            end
        end

        % Average tSC
        subplot(3,tSC_num + 1,tSC)
        title(sprintf('tsc%0.0f',tSC));
        plot(mean(sumemd),'LineWidth',2,'Color',Colors(tSC,:));
        xlim([0,time_window * 2])
        set(gca,'xtick',[1,time_window,time_window * 2])
        set(gca,'xticklabel',({'-100','0','100'}))
        set(gca,'ytick',[])
        ylabel('average tsc')

        % tSC triggered raster plots
        Ra=subplot(3,tSC_num + 1,tSC + (tSC_num + 1));
        title('cell firing triggered by the troughs of tSC signals')
        rasterplot(raszt',-time_window + 0.5:time_window - 0.5,Ra)
        ylabel('theta cycles')
        xlim([-time_window,time_window])
        set(gca,'xtick',[-time_window,0,time_window])
        set(gca,'xticklabel',({'-100','0','100'}))

        % Psth and Z-shift
        subplot(3,tSC_num + 1,tSC + (tSC_num + 1)*2)
        kernel_time = -kernels(tSC):kernels(tSC);
        kernel = normpdf(kernel_time,0,2*kernels(tSC));       % define smoothing kernel
        spsth = conv(mean(raszt,2),kernel)*Samplingrate;
        plot(spsth(kernels(tSC)+1:end-kernels(tSC)))
        xlabel({'time from trough of' ; 'the tSC signal (ms)'})
        ylabel('blue: firing rate (Hz); red: Z')
        set(gca,'xtick',[1,time_window,time_window * 2])
        set(gca,'xticklabel',({'-100','0','100'}))
        hold on
        yyaxis right
        plot(linspace(1,time_window * 2,time_window * 2 + 1),Zshift_tsc(tSC,:))

    end

    % Save figure
    if ~isshuffle
        saveas(gcf, fullfile(base, figfold, [filename,'_', ID,'.png']));
    end
    % Z-shift p value correction for statistical power differences
    % between tSCs (due to different spike numbers)
    p_corr=NaN(4,time_window * 2 + 1);
    for tsc = 1:4
        for tau_c = 1:time_window * 2 + 1
            try
                if length(phaserel2tsc_Z_t{tsc}{tau_c}) > min_spikes
                    [~,p_corr(tsc,tau_c),~,~] = b_rao(phaserel2tsc_Z_t{tsc}{tau_c}(randperm(length(phaserel2tsc_Z_t{tsc}{tau_c}),min_spikes)));
                else
                    [~,p_corr(tsc,tau_c),~,~] = b_rao(phaserel2tsc_Z_t{tsc}{tau_c});
                end
            catch
                p_corr(tsc,tau_c) = NaN;
            end
        end
    end

    % Find the current cell
    if isempty(Matrix)
        Matrix(1).ID = [filename,'_', ID];
    end
    M_index = find(strcmp({Matrix.ID}, [filename,'_', ID]) == 1);
    if isempty(M_index)
        M_index = length(Matrix) + 1;
        Matrix(M_index).ID = [filename,'_', ID]; % if not yet in the Matrix add
    end

    % Add/update results to the Matrix
    Matrix(M_index).mainfreqs = mainfreqs;
    Matrix(M_index).Zshift_tsc = Zshift_tsc;
    Matrix(M_index).pRayleigh_Zshift_tsc = pRayleigh_Zshift_tsc;
    Matrix(M_index).phaserel2tsc_A = phaserel2tsc;
    Matrix(M_index).p_rayleigh_corr = p_corr;

end

% Save results
save(fullfile(mainPath,[Matrixname,'.mat']),'Matrix');

