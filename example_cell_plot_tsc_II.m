function example_cell_plot_tsc_II(ID,tWindow1,tWindow2)
%EXAMPLE_CELL_PLOT_TSC_II Demonstration of a rhythmicity group member.
%   EXAMPLE_CELL_PLOT_TSC_II(ID,TWINDOW) plots the activity of the given cell
%   (ID) from a recording between TWINDOW(1) and TWINDOW(2) seconds; and
%   also the the autocorrelation of it. It also shows all the acgs of the
%   ryhthmicity group during theta and delta and the mean acgs.
%   Parameters:
%   IDS: 1x4 cell array containing the ID (animalId, recordingId, shankId,
%   cellId) of the MS cell (e.g. {20161989,163164,2,64});
%   or a number containing rowId (in allCell matrix).
%   TWINDOW: 1x2 vector, (e.g. [1879,1886]).
%
%   See also EXAMPLE_CELLS_PLOT, CONVERT_IDS, HIPPO_STATE_DETECTION,
%   PLOT_HIPPO_AND_CELLS.

%   Author: Barnabas Kocsis
%   Institute of Experimental Medicine, MTA
%   Date: 29/01/2021

global RESULTDIR
global CGWINDOW
global NSR

clear all
defaultDataRoot = '/Volumes/T7_Taka/Minnesota/MSHC/Data_VargaV/23798184';
defaultBrowseRoot = '/Volumes/T7_Taka/Minnesota/MSHC/Data_VargaV';
dataRoot = defaultDataRoot;
if ~isfolder(dataRoot)
    browseRoot = defaultBrowseRoot;
    if ~isfolder(browseRoot)
        browseRoot = pwd;
    end
    dataRoot = uigetdir(browseRoot,'Select data root folder (23798184)');
    if isequal(dataRoot,0)
        error('No data root folder selected.');
    end
end
paramFile = fullfile(dataRoot,'parameters.mat');
if exist(paramFile,'file')
    load(paramFile);
else
    [paramName,paramDir] = uigetfile('*.mat','Select parameters.mat',dataRoot);
    if isequal(paramName,0)
        error('No parameters file selected.');
    end
    load(fullfile(paramDir,paramName));
end

savePath = uigetdir(dataRoot,'Select folder for saving outputs');
if isequal(savePath,0)
    savePath = dataRoot;
end

grpId = 'pacemakers';
ID = {20161989,151152,2,9};
tWindow1 = [1972.5,1975.5]; tWindow2 = [1533,1536];
cLims = [0.000149744570807004,0.000245071370351526];
h = 16.7073123727004;

% tWindow1 = [1970,1976]; % tWindow2 = [1440,1446]; % tWindow2 = [1460,1466];

% grpId = 'tonic';
% ID = {20161989,113114,3,12};
% tWindow1 = [131,134]; tWindow2 = [965,968];
% cLims = [1.43899977181575e-05,0.000234811756898123];
% h = 14.1734560615832;

% tWindow1 = [129,135]; % tWindow2 = [961,967];

% grpId = 'theta_followers';
% ID = {20161989,163164,2,64};
% tWindow1 = [2249,2252]; tWindow2 = [584,587];
% cLims = [0.00016082764513059,0.000228743223857318];
% h =  14.7018116476572;

% tWindow1 = [2241,2244]; tWindow2 = [584,587]; % tWindow1 = [2240,2246]; % tWindow2 = [675,681]; % tWindow2 = [687,693];

% Parameters for adjusting the acg windows:
maxLag = 0.5; % maximum lag (sec)
xLim = (CGWINDOW-maxLag)*NSR;

% Load data table
load(fullfile(RESULTDIR, 'cell_features','allCell.mat'), 'allCell');
% Load map for allCell matrix (mO):
load(fullfile(RESULTDIR, 'cell_features','allCellMap.mat'),'mO');
% Load rhythmicity group abbreviations:
load(fullfile(RESULTDIR, 'rhythmic_groups','rhGroups.mat'),'rhGroups');

if nargin == 0
    variable_definitions; %IDs,tWindow definitions
end

if iscell(ID)
    rowID = find_rowIds(ID{1},ID{2},ID{3},ID{4});
else
    rowID = ID;
end

animalId = num2str(allCell(rowID,mO('animalId')));
recordingId = num2str(allCell(rowID,mO('recordingId')));
shankId = allCell(rowID, mO('shankId'));
cellId = allCell(rowID, mO('cellId'));

% F = figure('Position',get(0,'Screensize'));
F = figure('Position',[1,40,1500,500]);

%% MS unit raster plot and hippocampal data:
% Theta:
subplot(3,4,[1,2]);
% figure;
raster_plot(rowID,tWindow1);
xLims = xlim; hold on, plot([xLims(1),xLims(1)],[0,h]/5,'k','LineWidth',2) % 0.2 mV bar
% set(gca,'Visible','off'), savefig(fullfile(savePath,'raster_theta',grpId)); close

% Delta:
subplot(3,4,[3,4]);
% figure;
raster_plot(rowID,tWindow2);
xLims = xlim; hold on, plot([xLims(1),xLims(1)],[0,h]/5,'k','LineWidth',2) % 0.2 mV bar
% set(gca,'Visible','off'), savefig(fullfile(savePath,'raster_delta',grpId)); close

%% Open desired septal cell's acg:
subplot(3,4,[5,9]);
% figure
hold on, title([num2str(shankId),' ',num2str(cellId)])
plot(allCell(rowID,mO('thetaAcgFirst')+xLim:mO('thetaAcgLast')-xLim))
plot(allCell(rowID,mO('deltaAcgFirst')+xLim:mO('deltaAcgLast')-xLim))
set(gca,'xtick',[0,maxLag*NSR,maxLag*NSR*2]); set(gca,'xticklabel',[-maxLag,0,maxLag]); xlabel('Lag (s)')
ylim([0,4]*1e-4)
% savefig(fullfile(savePath,'example_acg',grpId)); close

%% Open rythmicity group's figures:
rhGroupId = rhGroups{allCell(rowID,mO('rhGroup'))}; % find out rhythmicity group abbr.
rowIds = get_rhGroup_indices_in_allCell(rhGroupId);

% Sort acgs based on theta-index-during-theta:
sortedCells = sortrows(allCell(rowIds,:),mO('ThAcgThInx'));

% Population acgs (theta):
subplot(3,4,[6,10]);
% figure
imageccgs(flipud(sortedCells(:,mO('thetaAcgFirst')+xLim:mO('thetaAcgLast')-xLim))); colormap jet
title('During theta');
set(gca,'xtick',[0,maxLag*NSR,maxLag*NSR*2]+0.5); set(gca,'xticklabel',[-maxLag,0,maxLag]); xlabel('Lag (s)')
colormap parula
caxis(cLims)
% savefig(fullfile(savePath,'population_acg_theta',grpId)); close
% writematrix(flipud(sortedCells(:,mO('thetaAcgFirst')+xLim:mO('thetaAcgLast')-xLim)),fullfile(savePath,'population_acg_theta',[grpId,'-theta.xlsx']));

% Population acgs (delta):
subplot(3,4,[7,11]);
% figure
imageccgs(flipud(sortedCells(:,mO('deltaAcgFirst')+xLim:mO('deltaAcgLast')-xLim))); colormap jet
title('During delta');
set(gca,'xtick',[0,maxLag*NSR,maxLag*NSR*2]+0.5); set(gca,'xticklabel',[-maxLag,0,maxLag]); xlabel('Lag (s)')
colormap parula
caxis(cLims), colorbar
% savefig(fullfile(savePath,'population_acg_delta',grpId)); close
% writematrix(flipud(sortedCells(:,mO('deltaAcgFirst')+xLim:mO('deltaAcgLast')-xLim)),fullfile(savePath,'population_acg_delta',[grpId,'-delta.xlsx']));

% Mean acgs:
subplot(3,4,[8,12]);
% figure
mThAcg = mean(allCell(rowIds,mO('thetaAcgFirst')+xLim:mO('thetaAcgLast')-xLim)); % mean theta acg
mDeAcg = mean(allCell(rowIds,mO('deltaAcgFirst')+xLim:mO('deltaAcgLast')-xLim)); % mean delta acg
plot(mThAcg), hold on, plot(mDeAcg)
title('Mean acg');
set(gca,'xtick',[0,maxLag*NSR,maxLag*NSR*2]); set(gca,'xticklabel',[-maxLag,0,maxLag]); xlabel('Lag (s)')
ylim([0,4]*1e-4)
% savefig(fullfile(savePath,'mean_acg',grpId)); close

% set(gcf,'Position',[1,41,1500,300]);
% saveas(gcf,fullfile(savePath,[grpId,'.png']));
end
