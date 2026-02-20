function neuron_spectra(mainPath, animal, session, ch, isstim, neuron_n)
%NEURON_SPECTRA Spectral analysis of spike trian data
%   NEURON_SPECTRA(MAINPATH, ANIMAL, SESSION, CH, ISSTIM, NEURON_N)
%   transforms spike train data to a pseudo-continous signal, and computes
%   the Furier spectrum, the burst triggered average wavelet spectrum and
%   the brust triggered average coherence with the CA1 LFP.

%   Required input arguments:
%       MAINPATH: the acces route of the folder containing the results
%           matrix and the database: folders with animal IDs containing
%           folders with session IDs, containing the preprocessed data
%           (clustered spike times, and extracted tSCs in the \raw folder).
%       ANIMAL: ID of the animal
%       SESSION: ID of the session
%       CH: number of the LFP channel used for choerence analysis
%       ISSTIM: logical variable indicating whether there are stimulation
%       periods that need to be exluded from the data.
%   Optional input arguments:
%       NEURON_N: if only one neuron should be examined, the number of the
%       neuron

%   Bálint Király
%   Institute of Experimental Medicine, Budapest, Hungary
%   kiraly.balint@koki.hu
%   19-Dec-2021

% Settings
global Samplingrate;
kernel_size = 8;
time_window = 100;

% Define directories
figfold = 'Spectra';
base = fullfile(mainPath,animal,session);
folder = 'raw';
filename = [animal,session];
fullpath = fullfile(base,folder,[filename,'_1']);
mkdir(fullfile(base,figfold))
Matrixname = 'Matrix';

% Load results Matrix
if exist(fullfile(mainPath,[Matrixname,'.mat']), 'file')
    Matrix = load(fullfile(mainPath,[Matrixname,'.mat']));
else
    Matrix = struct('Matrix',struct('ID',{})); % create Matrix if not exist
end
Matrix = Matrix.Matrix;

% Load LFP and stimulation times
eeg = cell2mat(struct2cell(load([fullpath,'.eeg.', ch ,'.mat'])));
if isstim
    stim = cell2mat(struct2cell(load(fullfile(mainPath,'STIMULATIONS',[filename,'.mat']),'stim')));
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

    % Omit spikes during stimulation period
    if isstim == 1
        spikes(stim(spikes) == 1) = [];
    end

    % Compute pseudo-continous spike train signal
    kernel = gausswin(kernel_size);
    spiketrain = zeros(1,(length(eeg)));
    spiketrain(spikes) = 1;
    spsth = conv(spiketrain,kernel);
    spsth = spsth(ceil(length(kernel)/2):end-floor(length(kernel)/2));

    % Perform FFT
    N = length(spsth);
    FFT_Spect = fft(spsth);
    Spect2 = abs(FFT_Spect/N); % two-sided spectrum
    Spect1 = Spect2(1:N/2+1); % single-sided spectrum
    Spect1(2:end-1) = 2*Spect1(2:end-1);
    freqs = Samplingrate*(0:(N/2))/N;
    [~,ind4Hz]=min(abs(freqs-4));
    [~,ind1Hz]=min(abs(freqs-1));
    [~,ind200Hz]=min(abs(freqs-200));
    smooth_pow = movmean(Spect1,ceil(length(freqs)/Samplingrate*2)); % smoothing spectrum
    % normalize spectrum with the area under the curve in the 4-200 Hz range
    smooth_pow = smooth_pow / sum(smooth_pow(ind4Hz:ind200Hz));

    % Detect bursts
    [burstWind] = burst_detector(spikes,40,spikes(end));
    burst_start = find(diff([0 burstWind])==1);

    % Plot the Burst-start triggered average of the pseudo-continous
    % spike train signal
    figure
    ETA(burst_start,spsth,Samplingrate,time_window,1,1);
    saveas(gcf, fullfile(base, figfold, [filename,'_', ID,'BTAS.png']));

    % Plot the Burst-start triggered average coherence between the pseudo-continous
    % spike train signal and the LFP signal
    figure
    ETA_cross(burst_start,spsth,eeg,Samplingrate,time_window,1,1);
    saveas(gcf, fullfile(base, figfold, [filename,'_', ID,'BTAC.png']));

    % Find the current cell in the Matrix
    M_index = find(strcmp({Matrix.ID}, [filename,'_', ID]) == 1);
    if isempty(M_index)
        M_index = length(Matrix) + 1;
        Matrix(M_index).ID = [filename,'_', ID]; % if not yet in the Matrix add
    end

    % Add/update results to the Matrix
    Matrix(M_index).smooth_pow = smooth_pow (ceil(linspace(ind1Hz,ind200Hz,200)));

end

% Save results
save(fullfile(mainPath,[Matrixname,'.mat']),'Matrix');
