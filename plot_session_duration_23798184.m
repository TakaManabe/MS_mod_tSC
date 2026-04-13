function sessionLenTable = plot_session_duration_23798184(dataRoot, varargin)
%PLOT_SESSION_DURATION_23798184 Build and plot per-session signal lengths.
%   SESSIONLENTABLE = PLOT_SESSION_DURATION_23798184(DATAROOT)
%   scans session folders (same discovery logic as directional scripts),
%   estimates HC signal length per session, and writes:
%     - session_length_inventory.csv
%     - fig_session_length_by_group_samples.png
%     - fig_session_length_by_group_minutes.png
%
%   Name-value options:
%     'OutputDir'     : output folder (default DATAROOT/analysis_directional)
%     'SamplingRate'  : Hz for seconds/minutes conversion (default 1000)
%     'StimRoot'      : stim dataset root(s), char or cellstr (optional)
%     'ControlRoot'   : control dataset root(s), char or cellstr (optional)
%     'Verbose'       : print progress (default true)

ip = inputParser;
ip.addRequired('dataRoot', @is_text_scalar);
ip.addParameter('OutputDir', '', @is_text_scalar);
ip.addParameter('SamplingRate', 1000, @isscalar);
ip.addParameter('StimRoot', '', @is_text_or_cellstr);
ip.addParameter('ControlRoot', '', @is_text_or_cellstr);
ip.addParameter('Verbose', true, @islogical);
ip.parse(dataRoot, varargin{:});
opt = ip.Results;

dataRoot = as_char(dataRoot);
opt.OutputDir = as_char(opt.OutputDir);
if isempty(opt.OutputDir)
    opt.OutputDir = fullfile(dataRoot, 'analysis_directional');
end

stimRoots = to_cellstr(opt.StimRoot);
controlRoots = to_cellstr(opt.ControlRoot);

if isempty(stimRoots) && isempty(controlRoots)
    [stimRoots, controlRoots] = autodetect_roots(dataRoot);
end

if ~isfolder(opt.OutputDir)
    mkdir(opt.OutputDir);
end

if opt.Verbose
    fprintf('Data root: %s\n', dataRoot);
    fprintf('Output   : %s\n', opt.OutputDir);
    fprintf('Fs (Hz)  : %.6g\n', opt.SamplingRate);
end

[~, sessionTable] = print_stim_control_inventory(dataRoot, ...
    'StimRoot', stimRoots, ...
    'ControlRoot', controlRoots, ...
    'OutputDir', opt.OutputDir, ...
    'Verbose', false, ...
    'SaveTables', true);

if isempty(sessionTable)
    warning('plot_session_duration_23798184:NoSessions', ...
        'No sessions found from specified roots.');
    sessionLenTable = table();
    return;
end

if opt.Verbose
    fprintf('Sessions found: %d\n', height(sessionTable));
end

n = height(sessionTable);
nSamples = nan(n, 1);
durSec = nan(n, 1);
durMin = nan(n, 1);
sigSource = repmat({''}, n, 1);
status = repmat({'ok'}, n, 1);
message = repmat({''}, n, 1);

for i = 1:n
    dataDir = sessionTable.data_dir{i};
    if ~isfolder(dataDir)
        status{i} = 'skip';
        message{i} = 'data_dir missing';
        continue;
    end

    [L, src] = estimate_session_length(dataDir);
    if isnan(L) || L <= 0
        status{i} = 'skip';
        message{i} = 'no loadable HC signal';
        continue;
    end

    nSamples(i) = L;
    durSec(i) = L / opt.SamplingRate;
    durMin(i) = durSec(i) / 60;
    sigSource{i} = src;
end

sessionLenTable = sessionTable(:, {'dataset','condition','animal','session','layout','base_dir','data_dir','n_hc_channels','n_ms_tetrodes'});
sessionLenTable.n_samples = nSamples;
sessionLenTable.duration_sec = durSec;
sessionLenTable.duration_min = durMin;
sessionLenTable.signal_source = sigSource;
sessionLenTable.status = status;
sessionLenTable.message = message;

csvPath = fullfile(opt.OutputDir, 'session_length_inventory.csv');
writetable(sessionLenTable, csvPath);

valid = strcmp(sessionLenTable.status, 'ok') & isfinite(sessionLenTable.n_samples) & sessionLenTable.n_samples > 0;
if ~any(valid)
    warning('plot_session_duration_23798184:NoLengths', ...
        'No session length could be computed. CSV saved: %s', csvPath);
    return;
end

plotT = sessionLenTable(valid, :);
group = strcat(plotT.dataset, ' | ', plotT.condition);
group = categorical(group, unique(group, 'stable'));

fig1 = figure('Color', 'w', 'Position', [100 100 1200 520]);
subplot(1, 2, 1);
draw_distribution(group, double(plotT.n_samples), 'Session length (samples)');
title('Per-session signal length');
subplot(1, 2, 2);
draw_distribution(group, double(plotT.duration_min), 'Session length (minutes)');
title(sprintf('Per-session signal length (Fs=%.6g Hz)', opt.SamplingRate));

png1 = fullfile(opt.OutputDir, 'fig_session_length_by_group_samples_minutes.png');
saveas(fig1, png1);

fig2 = figure('Color', 'w', 'Position', [100 100 920 520]);
datasets = categorical(plotT.dataset, unique(plotT.dataset, 'stable'));
draw_distribution(datasets, double(plotT.duration_min), 'Session length (minutes)');
title('Per-session length by dataset');
png2 = fullfile(opt.OutputDir, 'fig_session_length_by_dataset_minutes.png');
saveas(fig2, png2);

if opt.Verbose
    fprintf('Saved: %s\n', csvPath);
    fprintf('Saved: %s\n', png1);
    fprintf('Saved: %s\n', png2);
    print_group_summary(plotT);
end
end

function draw_distribution(groups, y, yLabelTxt)
ax = gca;
hold(ax, 'on');
if exist('violinplot', 'file') == 2
    try
        violinplot(y, groups, 'ShowData', false, 'ShowMean', true);
        ylabel(yLabelTxt);
        xtickangle(25);
        grid on;
        hold(ax, 'off');
        return;
    catch
        % fallback below
    end
end

if exist('boxchart', 'file') == 2
    x = double(groups);
    boxchart(x, y, 'BoxFaceColor', [0.65 0.75 0.90], 'MarkerStyle', 'none');
    jitter = (rand(size(x)) - 0.5) * 0.24;
    scatter(x + jitter, y, 12, 'k', 'filled', 'MarkerFaceAlpha', 0.35, 'MarkerEdgeAlpha', 0.35);
    set(ax, 'XTick', 1:numel(categories(groups)), 'XTickLabel', categories(groups));
else
    boxplot(y, groups, 'Symbol', '');
    x = double(groups);
    jitter = (rand(size(x)) - 0.5) * 0.24;
    scatter(x + jitter, y, 12, 'k', 'filled', 'MarkerFaceAlpha', 0.35, 'MarkerEdgeAlpha', 0.35);
end

ylabel(yLabelTxt);
xtickangle(25);
grid on;
hold(ax, 'off');
end

function [stimRoots, controlRoots] = autodetect_roots(dataRoot)
dirs = list_immediate_subdirs(dataRoot);
dirs = dirs(~strcmpi(dirs, 'Zip'));
roots = cellfun(@(d) fullfile(dataRoot, d), dirs, 'UniformOutput', false);

isStim = false(size(roots));
for i = 1:numel(roots)
    isStim(i) = has_stim_evidence(roots{i});
end

stimRoots = roots(isStim);
controlRoots = roots(~isStim);
if isempty(controlRoots)
    controlRoots = roots;
end
end

function [L, src] = estimate_session_length(dataDir)
L = nan;
src = '';

eegFiles = valid_files(dir(fullfile(dataDir, '*.eeg.*.mat')));
bestLen = -1;
bestSrc = '';
for i = 1:numel(eegFiles)
    f = fullfile(dataDir, eegFiles(i).name);
    v = load_first_numeric_vector(f);
    if isempty(v)
        continue;
    end
    nv = numel(v);
    if nv > bestLen
        bestLen = nv;
        bestSrc = f;
    end
end
if bestLen > 0
    L = bestLen;
    src = bestSrc;
    return;
end

regionPat = {'*radiatum*.mat', '*oriens*.mat', '*pyramid*.mat', '*ca1*.mat', '*ca3*.mat', '*dg*.mat'};
for pi = 1:numel(regionPat)
    files = valid_files(dir(fullfile(dataDir, regionPat{pi})));
    for i = 1:numel(files)
        f = fullfile(dataDir, files(i).name);
        v = load_first_numeric_vector(f);
        if isempty(v)
            continue;
        end
        L = numel(v);
        src = f;
        return;
    end
end
end

function print_group_summary(T)
fprintf('\n=== Session length summary (minutes) ===\n');
groups = unique(strcat(T.dataset, ' | ', T.condition), 'stable');
for i = 1:numel(groups)
    idx = strcmp(strcat(T.dataset, ' | ', T.condition), groups{i});
    x = T.duration_min(idx);
    x = x(isfinite(x));
    if isempty(x)
        fprintf('%s : n=0\n', groups{i});
    else
        fprintf('%s : n=%d, median=%.2f, IQR=[%.2f, %.2f]\n', ...
            groups{i}, numel(x), median(x), prctile(x, 25), prctile(x, 75));
    end
end
end

function tf = has_stim_evidence(datasetRoot)
tf = false;
[~, dsName] = fileparts(datasetRoot);
nameLower = lower(dsName);
if ~isempty(strfind(nameLower, 'stim'))
    tf = true;
    return;
end
if isfolder(fullfile(datasetRoot, 'STIMULATIONS'))
    tf = true;
    return;
end

d = dir(fullfile(datasetRoot, '**', 'StimEvents.mat'));
if ~isempty(d)
    tf = true;
    return;
end

d2 = dir(fullfile(datasetRoot, '**', '*stim*.mat'));
if ~isempty(d2)
    tf = true;
end
end

function dirs = list_immediate_subdirs(parent)
d = dir(parent);
d = d([d.isdir]);
dirs = {};
for i = 1:numel(d)
    nm = d(i).name;
    if strcmp(nm, '.') || strcmp(nm, '..')
        continue;
    end
    if strncmp(nm, '._', 2)
        continue;
    end
    dirs{end + 1} = nm; %#ok<AGROW>
end
dirs = sort(dirs);
end

function vec = load_first_numeric_vector(matFile)
vec = [];
try
    s = load(matFile);
catch
    return;
end
fn = fieldnames(s);
for i = 1:numel(fn)
    x = s.(fn{i});
    if isnumeric(x)
        x = x(:);
        if ~isempty(x)
            vec = x;
            return;
        end
    end
end
end

function files = valid_files(d)
if isempty(d)
    files = d;
    return;
end
keep = false(size(d));
for i = 1:numel(d)
    nm = d(i).name;
    keep(i) = ~d(i).isdir && ~strncmp(nm, '._', 2);
end
files = d(keep);
end

function out = to_cellstr(x)
if isempty(x)
    out = {};
    return;
end
if ischar(x)
    out = {x};
    return;
end
if isstring(x)
    out = cellstr(x(:));
    return;
end
if iscell(x)
    out = cell(size(x));
    for i = 1:numel(x)
        out{i} = as_char(x{i});
    end
    return;
end
error('Expected char/string/cellstr.');
end

function tf = is_text_scalar(x)
tf = ischar(x) || (isstring(x) && isscalar(x));
end

function tf = is_text_or_cellstr(x)
tf = is_text_scalar(x) || iscellstr(x) || (isstring(x) && ~isscalar(x)) || isempty(x);
end

function s = as_char(x)
if ischar(x)
    s = x;
elseif isstring(x) && isscalar(x)
    s = char(x);
else
    error('Expected text scalar.');
end
end
