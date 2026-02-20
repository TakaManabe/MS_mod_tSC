function Z_decr = tSC_coupling_shufflingcontroll(mainpath,p_sig,minspikesnum,tSC_num)
%TSC_COUPLING_SHUFFLINGCONTROLL Shuffling controll of tSC coupling.
%   Z_DECR = TSC_COUPLING_SHUFFLINGCONTROLL[MAINPATH,P_SZIG,MINSPIKESNUM,TSC_NUM]
%   Compares tSC coupling of real spikes vs shuffled spikes at P_SIG
%   significance level at any of the temporal shifts pretested with the
%   TSC_NEURON_COUPLING function. Nerons with less spikes than
%   MINSPIKESNUM during theta cycles with tSC are automatically considered
%   non coupled. TSC_NUM defines the number of tSC required to be tested.
%   The Figure S6 of the Király et al. manuscript and he number of
%   neurons with decreased maximal Rayleigh's Z is returned for each tSC
%   (Z_DECR).
%
%   See also TSC_NEURON_COUPLING

%   Bálint Király
%   Institute of Experimental Medicine, Budapest, Hungary
%   kiraly.balint@koki.hu
%   01-Jan-2022


% Load preprocessed non-shuffled and shuffled coupling data
load(fullfile(mainpath,'Matrix.mat'),'Matrix');
Matrix_shuffle = load(fullfile(mainpath,'Matrix_shuffle.mat'),'Matrix');
Matrix_shuffle = Matrix_shuffle.Matrix;
n = length(Matrix);
% Find significcant tSC coupling for non-shuffled and shuffled spikes
significance_matrix = tSC_coupling_test(Matrix,p_sig,minspikesnum,ones(1,n),tSC_num);
significance_matrix_shuffle = tSC_coupling_test(Matrix_shuffle,p_sig,minspikesnum,ones(1,n),tSC_num);

% Find maximal Z-values for each neuron-tSC pair for both shuffled and
% non-shuffled spikes
Z_max_shuffle = NaN(n,tSC_num+1);
Z_max = NaN(n,tSC_num+1);
for neuron = 1:n
    Z_max_shuffle(neuron,:) = max(Matrix_shuffle(neuron).Zshift_tsc,[],2);
    Z_max(neuron,:) = max(Matrix(neuron).Zshift_tsc,[],2);
end

% Panel S6A
figure
hold on
Z_decr = zeros(1,tSC_num);
for tSC = 1:tSC_num
    subplot(1,tSC_num,tSC)
    line([1,2],[squeeze(Z_max(significance_matrix(tSC,:) == 1,tSC)),squeeze(Z_max_shuffle(significance_matrix(tSC,:) == 1,tSC))],'Color',[0,0,0,0.05])
    Z_decr(tSC) = sum(Z_max(significance_matrix(tSC,:) == 1,tSC) < Z_max_shuffle(significance_matrix(tSC,:) == 1,tSC));
    line([1,2],[mean(squeeze(Z_max(significance_matrix(tSC,:) == 1,tSC))),mean(squeeze(Z_max_shuffle(significance_matrix(tSC,:) == 1,tSC)))],'Color',[0,0,0])
    xlim([0,3])
end


% Panel S6B
figure
bar(sum(significance_matrix,2))
hold on
bar(sum(((significance_matrix + significance_matrix_shuffle) == 2),2),'w')
ylabel('Number of neurons')
setmyplot_balazs;
