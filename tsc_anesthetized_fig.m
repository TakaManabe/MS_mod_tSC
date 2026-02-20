function tsc_anesthetized_fig(mainpath)
%TSC_ANESTHETIZED Main wrapper for the figure 6 presented in the Király
%   et al. manuscript.
%
%   Required input arguments:
%       MAINPATH: Fullpath of the database and results matrix

%   Bálint Király
%   Institute of Experimental Medicine, Budapest, Hungary
%   kiraly.balint@koki.hu
%   23-Dec-2021

% Panel F6A and S12A is created in python with the plotMeantSCcycle
% function of the 'Analysis framework for the extraction of theta-nested
% spectral components' package by Lopes-dos-Santos et al., 2018.

% load data
load(fullfile(mainpath,'ses_Matrix.mat'));
load(fullfile(mainpath,'Matrix.mat'));
n_ses = length(ses_Matrix);

% Panel B
tSC_num = 5;
tSC_spect = horzcat(ses_Matrix.tSCs);
tSC_spect = reshape(tSC_spect,[size(tSC_spect,1),tSC_num,n_ses]);
tSC_spectrum_plotter(tSC_spect,1:n_ses);

% Panel C
numberofcycles = vertcat(ses_Matrix.numberofcycles);
numberofallcycles = vertcat(ses_Matrix.numberofallcycles);
numberofcycles_norm = bsxfun(@rdivide,numberofcycles,numberofallcycles);
figure
subplot(2,1,1)
tsc_barplot(numberofcycles_norm(:,1:tSC_num));
subplot(2,1,2)
tSC_ratio = (numberofallcycles - numberofcycles(:,end))./numberofallcycles;
tsc_barplot([tSC_ratio,1-tSC_ratio]);

% Find significcant tSC coupling
tSC_num = 4;
p_sig = 0.05;
minspikesnum = 10;
n = length(Matrix);
significance_matrix = tSC_coupling_test(Matrix,p_sig,minspikesnum,ones(1,n),tSC_num);

% Panel D
tSC_population_psth(Matrix,significance_matrix,tSC_num,160,12);
tSC_coupled_neurons_distribution(significance_matrix,tSC_num,vertcat(Matrix.pRayleigh),vertcat(Matrix.rythmgroupnum));
