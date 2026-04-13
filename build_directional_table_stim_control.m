function [eventTable, sessionTable] = build_directional_table_stim_control(dataRoot, varargin)
%BUILD_DIRECTIONAL_TABLE_STIM_CONTROL Build event table for directional analysis.
%   [EVENTTABLE, SESSIONTABLE] = BUILD_DIRECTIONAL_TABLE_STIM_CONTROL(DATAROOT)
%   builds a per-event table from STIM_Exp + awake_mouse_control datasets.
%
%   Features:
%   - HC pre/post amplitude and theta power (4-12 Hz)
%   - MS spike counts/rates from TT*.mat units
%   - condition labels: 'stim' or 'control'
%
%   Name-value options:
%       'StimRoot'            : STIM dataset root path
%       'ControlRoot'         : control dataset root path
%       'OutputDir'           : output folder (default DATAROOT)
%       'SamplingRate'        : Hz (default 1000)
%       'WindowMs'            : pre/post window in ms (default 200)
%       'MinInterEventMs'     : merge close stim events (default 300)
%       'ControlEventsPerSession' : fixed pseudo event count for control
%                                   (default: median stim event count, else 200)
%       'RngSeed'             : random seed (default 42)
%       'Verbose'             : print progress (default true)
%
%   Output files:
%       directional_event_table_stim_control.csv
%       directional_event_table_stim_control.mat

ip = inputParser;
ip.addRequired('dataRoot', @is_text_scalar);
ip.addParameter('StimRoot', '', @is_text_or_cellstr);
ip.addParameter('ControlRoot', '', @is_text_or_cellstr);
ip.addParameter('OutputDir', '', @is_text_scalar);
ip.addParameter('SamplingRate', 1000, @isscalar);
ip.addParameter('WindowMs', 200, @isscalar);
ip.addParameter('MinInterEventMs', 300, @isscalar);
ip.addParameter('ControlEventsPerSession', [], @(x) isempty(x) || isscalar(x));
ip.addParameter('RngSeed', 42, @isscalar);
ip.addParameter('Verbose', true, @islogical);
ip.parse(dataRoot, varargin{:});
opt = ip.Results;

dataRoot = as_char(dataRoot);
stimRoots = to_cellstr(opt.StimRoot);
controlRoots = to_cellstr(opt.ControlRoot);
opt.OutputDir = as_char(opt.OutputDir);

if isempty(opt.OutputDir)
    opt.OutputDir = dataRoot;
end

if isempty(stimRoots)
    stimRoots = {fullfile(dataRoot, 'STIM_Exp')};
end
if isempty(controlRoots)
    controlRoots = {fullfile(dataRoot, 'awake_mouse_control')};
end

[~, sessionTable] = print_stim_control_inventory(dataRoot, ...
    'StimRoot', stimRoots, ...
    'ControlRoot', controlRoots, ...
    'OutputDir', opt.OutputDir, ...
    'Verbose', false, ...
    'SaveTables', false);

if isempty(sessionTable)
    error('build_directional_table_stim_control:NoSessions', ...
        ['No STIM/control sessions found. Checked roots: ', ...
         strjoin(stimRoots, ' ; '), ' || ', strjoin(controlRoots, ' ; ')]);
end

rng(opt.RngSeed);
winSamples = round(opt.WindowMs * opt.SamplingRate / 1000);
minInterSamples = round(opt.MinInterEventMs * opt.SamplingRate / 1000);

stimIdx = find(strcmp(sessionTable.condition, 'stim'));
ctrlIdx = find(strcmp(sessionTable.condition, 'control'));

eventRows = {};
stimEventCounts = [];

for i = 1:numel(stimIdx)
    row = sessionTable(stimIdx(i), :);
    [rows, nEvents] = process_one_session(row, 'stim', winSamples, minInterSamples, opt, []);
    if ~isempty(rows)
        eventRows = [eventRows; rows]; %#ok<AGROW>
    end
    if nEvents > 0
        stimEventCounts(end + 1) = nEvents; %#ok<AGROW>
    end
end

if isempty(opt.ControlEventsPerSession)
    if ~isempty(stimEventCounts)
        targetCtrlEvents = round(median(stimEventCounts));
        targetCtrlEvents = max(targetCtrlEvents, 20);
    else
        targetCtrlEvents = 200;
    end
else
    targetCtrlEvents = opt.ControlEventsPerSession;
end

for i = 1:numel(ctrlIdx)
    row = sessionTable(ctrlIdx(i), :);
    [rows, ~] = process_one_session(row, 'control', winSamples, minInterSamples, opt, targetCtrlEvents);
    if ~isempty(rows)
        eventRows = [eventRows; rows]; %#ok<AGROW>
    end
end

if isempty(eventRows)
    warning('build_directional_table_stim_control:NoEvents', ...
        'No events extracted. Returning empty table.');
    eventTable = table();
    return;
end

eventTable = cell2table(eventRows, 'VariableNames', { ...
    'dataset', ...
    'condition', ...
    'animal', ...
    'session', ...
    'layout', ...
    'event_id', ...
    'event_sample', ...
    'hc_channel', ...
    'n_tt_units', ...
    'n_ms_tetrodes', ...
    'hc_pre_rms', ...
    'hc_post_rms', ...
    'hc_delta_rms', ...
    'hc_pre_theta_pow', ...
    'hc_post_theta_pow', ...
    'hc_delta_theta_pow', ...
    'ms_pre_spike_count', ...
    'ms_post_spike_count', ...
    'ms_delta_spike_count', ...
    'ms_pre_rate_total', ...
    'ms_post_rate_total', ...
    'ms_delta_rate_total', ...
    'ms_pre_rate_per_unit', ...
    'ms_post_rate_per_unit', ...
    'ms_delta_rate_per_unit'});

if ~isfolder(opt.OutputDir)
    mkdir(opt.OutputDir);
end
writetable(eventTable, fullfile(opt.OutputDir, 'directional_event_table_stim_control.csv'));
save(fullfile(opt.OutputDir, 'directional_event_table_stim_control.mat'), 'eventTable');

if opt.Verbose
    fprintf('Directional table rows: %d\n', height(eventTable));
    fprintf('Saved: %s\n', fullfile(opt.OutputDir, 'directional_event_table_stim_control.csv'));
end

end

function [rows, nRawEvents] = process_one_session(row, condition, winSamples, minInterSamples, opt, controlTarget)
rows = {};
nRawEvents = 0;

dataset = row.dataset{1};
animal = row.animal{1};
session = row.session{1};
layout = row.layout{1};
rootPath = row.root_path{1};
baseDir = row.base_dir{1};
dataDir = row.data_dir{1};

hcList = parse_number_list(row.hc_channel_ids{1});
if isempty(hcList)
    hcList = detect_hc_channels(dataDir);
end
if isempty(hcList)
    if opt.Verbose
        fprintf('[SKIP] %s %s/%s: no HC eeg channel\n', dataset, animal, session);
    end
    return;
end
hcChannel = hcList(1);

[eeg, hcChannel, fullpath] = load_eeg_signal(rootPath, animal, session, dataDir, hcChannel);
if isempty(eeg)
    if opt.Verbose
        fprintf('[SKIP] %s %s/%s: eeg not loadable\n', dataset, animal, session);
    end
    return;
end
eeg = double(eeg(:));
N = numel(eeg);

[spikeCells, nMsTetrodes] = load_spike_units(baseDir, dataDir);
nUnits = numel(spikeCells);

if strcmp(condition, 'stim')
    events = get_stim_events(rootPath, animal, session, baseDir, dataDir, N, opt.SamplingRate);
    nRawEvents = numel(events);
    events = enforce_min_inter_event(events, minInterSamples);
else
    events = generate_control_events(N, winSamples, controlTarget);
    nRawEvents = numel(events);
end

if isempty(events)
    if opt.Verbose
        fprintf('[SKIP] %s %s/%s: no events\n', dataset, animal, session);
    end
    return;
end

events = events(events > winSamples & events <= (N - winSamples));
if isempty(events)
    if opt.Verbose
        fprintf('[SKIP] %s %s/%s: events out of bounds\n', dataset, animal, session);
    end
    return;
end

thetaPow = nan(size(eeg));
if opt.SamplingRate > 30
    try
        [b, a] = butter(3, [4 12] / (opt.SamplingRate / 2), 'bandpass');
        theta = filtfilt(b, a, eeg);
        thetaPow = theta .^ 2;
    catch
        thetaPow(:) = nan;
    end
end

winSec = winSamples / opt.SamplingRate;
for ei = 1:numel(events)
    ev = events(ei);
    preIdx = (ev - winSamples):(ev - 1);
    postIdx = ev:(ev + winSamples - 1);

    preSeg = eeg(preIdx);
    postSeg = eeg(postIdx);

    hcPreRms = sqrt(mean(preSeg .^ 2));
    hcPostRms = sqrt(mean(postSeg .^ 2));
    hcDeltaRms = hcPostRms - hcPreRms;

    hcPreTheta = mean_no_nan(thetaPow(preIdx));
    hcPostTheta = mean_no_nan(thetaPow(postIdx));
    hcDeltaTheta = hcPostTheta - hcPreTheta;

    [preSpk, postSpk] = count_spikes(spikeCells, preIdx(1), preIdx(end), postIdx(1), postIdx(end));
    deltaSpk = postSpk - preSpk;

    preRateTotal = preSpk / winSec;
    postRateTotal = postSpk / winSec;
    deltaRateTotal = postRateTotal - preRateTotal;

    if nUnits > 0
        preRateUnit = preRateTotal / nUnits;
        postRateUnit = postRateTotal / nUnits;
        deltaRateUnit = postRateUnit - preRateUnit;
    else
        preRateUnit = NaN;
        postRateUnit = NaN;
        deltaRateUnit = NaN;
    end

    rows(end + 1, :) = { ... %#ok<AGROW>
        dataset, ...
        condition, ...
        animal, ...
        session, ...
        layout, ...
        ei, ...
        ev, ...
        hcChannel, ...
        nUnits, ...
        nMsTetrodes, ...
        hcPreRms, ...
        hcPostRms, ...
        hcDeltaRms, ...
        hcPreTheta, ...
        hcPostTheta, ...
        hcDeltaTheta, ...
        preSpk, ...
        postSpk, ...
        deltaSpk, ...
        preRateTotal, ...
        postRateTotal, ...
        deltaRateTotal, ...
        preRateUnit, ...
        postRateUnit, ...
        deltaRateUnit};
end

if opt.Verbose
    fprintf('[OK] %s %s/%s: events=%d ch=%d units=%d\n', ...
        dataset, animal, session, numel(events), hcChannel, nUnits);
end

% Keep fullpath variable used during debug and avoid warnings in old MATLAB.
if isempty(fullpath)
    fullpath = '';
end
end

function [eeg, hcChannel, fullpath] = load_eeg_signal(rootPath, animal, session, dataDir, hcChannel)
eeg = [];
fullpath = '';
chCandidates = unique([hcChannel, detect_hc_channels(dataDir)]);

for i = 1:numel(chCandidates)
    ch = chCandidates(i);
    [fp, ~] = resolve_session_fullpath(rootPath, animal, session, num2str(ch));
    eegFile = [fp, '.eeg.', num2str(ch), '.mat'];
    if exist(eegFile, 'file') ~= 2
        continue;
    end
    vec = load_first_numeric_vector(eegFile);
    if ~isempty(vec)
        eeg = vec;
        hcChannel = ch;
        fullpath = fp;
        return;
    end
end

% Fallback: parse first eeg file directly.
eegFiles = valid_files(dir(fullfile(dataDir, '*.eeg.*.mat')));
for i = 1:numel(eegFiles)
    nm = eegFiles(i).name;
    tok = regexp(nm, '^(.*)\.eeg\.(\d+)\.mat$', 'tokens', 'once');
    if isempty(tok)
        continue;
    end
    eegFile = fullfile(dataDir, nm);
    vec = load_first_numeric_vector(eegFile);
    if isempty(vec)
        continue;
    end
    eeg = vec;
    hcChannel = str2double(tok{2});
    fullpath = fullfile(dataDir, tok{1});
    return;
end

% Fallback: region-named HC files (e.g., *_radiatum.mat).
regionPat = {'*radiatum*.mat', '*oriens*.mat', '*pyramid*.mat', '*ca1*.mat', '*ca3*.mat', '*dg*.mat'};
for pi = 1:numel(regionPat)
    rfiles = valid_files(dir(fullfile(dataDir, regionPat{pi})));
    for ri = 1:numel(rfiles)
        f = fullfile(dataDir, rfiles(ri).name);
        vec = load_first_numeric_vector(f);
        if isempty(vec)
            continue;
        end
        eeg = vec;
        hcChannel = 1;
        fullpath = f;
        return;
    end
end
end

function [spikeCells, nMsTetrodes] = load_spike_units(baseDir, dataDir)
spikeCells = {};
[ttFiles, tetrodes] = detect_tt_files(baseDir, dataDir);
nMsTetrodes = numel(tetrodes);

for i = 1:numel(ttFiles)
    vec = load_first_numeric_vector(ttFiles{i});
    if isempty(vec)
        continue;
    end
    v = double(vec(:));
    v = v(~isnan(v));
    if isempty(v)
        continue;
    end
    spikeCells{end + 1} = v; %#ok<AGROW>
end
end

function events = get_stim_events(rootPath, animal, session, baseDir, dataDir, N, fs)
events = [];

% 1) Preferred: STIMULATIONS vector.
try
    stim = load_session_stim(rootPath, animal, session);
    stim = double(stim(:));
    if isempty(stim)
        return;
    end
    stim = stim > 0;
    if numel(stim) < N
        stim(end + 1:N) = false;
    elseif numel(stim) > N
        stim = stim(1:N);
    end
    edges = diff([0; stim]) > 0;
    events = find(edges);
    if ~isempty(events)
        return;
    end
catch
end

% 2) Fallback: StimEvents.mat with BurstOn or PulseOn (seconds).
cand = { ...
    fullfile(baseDir, 'StimEvents.mat'), ...
    fullfile(dataDir, 'StimEvents.mat')};
for i = 1:numel(cand)
    if exist(cand{i}, 'file') ~= 2
        continue;
    end
    s = load(cand{i});
    if isfield(s, 'BurstOn') && ~isempty(s.BurstOn)
        x = s.BurstOn;
    elseif isfield(s, 'PulseOn') && ~isempty(s.PulseOn)
        x = s.PulseOn;
    else
        continue;
    end
    x = double(x(:));
    x = x(~isnan(x) & x > 0);
    if isempty(x)
        continue;
    end
    events = round(x * fs);
    events = unique(events(events > 0 & events <= N));
    if ~isempty(events)
        return;
    end
end
end

function events = enforce_min_inter_event(events, minInterSamples)
if isempty(events)
    return;
end
events = sort(unique(events(:)'));
if numel(events) < 2
    return;
end
keep = true(size(events));
last = events(1);
for i = 2:numel(events)
    if events(i) - last < minInterSamples
        keep(i) = false;
    else
        last = events(i);
    end
end
events = events(keep);
end

function events = generate_control_events(N, winSamples, targetCount)
if isempty(targetCount) || targetCount <= 0
    targetCount = 200;
end
low = winSamples + 1;
high = N - winSamples;
if high <= low
    events = [];
    return;
end
events = randi([low, high], targetCount, 1);
events = sort(unique(events));
end

function [preSpk, postSpk] = count_spikes(spikeCells, preBeg, preEnd, postBeg, postEnd)
preSpk = 0;
postSpk = 0;
for i = 1:numel(spikeCells)
    s = spikeCells{i};
    preSpk = preSpk + sum(s >= preBeg & s <= preEnd);
    postSpk = postSpk + sum(s >= postBeg & s <= postEnd);
end
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

function nums = detect_hc_channels(dataDir)
nums = [];
files = valid_files(dir(fullfile(dataDir, '*.eeg.*.mat')));
for i = 1:numel(files)
    tok = regexp(files(i).name, '\.eeg\.(\d+)\.mat$', 'tokens', 'once');
    if ~isempty(tok)
        nums(end + 1) = str2double(tok{1}); %#ok<AGROW>
    end
end
if isempty(nums)
    regionPat = {'*radiatum*.mat', '*oriens*.mat', '*pyramid*.mat', '*ca1*.mat', '*ca3*.mat', '*dg*.mat'};
    for pi = 1:numel(regionPat)
        ff = valid_files(dir(fullfile(dataDir, regionPat{pi})));
        if ~isempty(ff)
            nums = 1;
            break;
        end
    end
end
nums = unique(nums);
end

function [ttFiles, tetrodes] = detect_tt_files(baseDir, dataDir)
f1 = valid_files(dir(fullfile(baseDir, 'TT*.mat')));
f2 = valid_files(dir(fullfile(dataDir, 'TT*.mat')));
allNames = {};
for i = 1:numel(f1)
    allNames{end + 1} = fullfile(baseDir, f1(i).name); %#ok<AGROW>
end
for i = 1:numel(f2)
    allNames{end + 1} = fullfile(dataDir, f2(i).name); %#ok<AGROW>
end
ttFiles = unique(allNames);

tetrodes = [];
for i = 1:numel(ttFiles)
    [~, nm, ext] = fileparts(ttFiles{i});
    fileName = [nm, ext];
    tok = regexp(fileName, '^TT(\d+)_\d+\.mat$', 'tokens', 'once');
    if isempty(tok)
        tok = regexp(fileName, '^TT(\d+)\.mat$', 'tokens', 'once');
    end
    if ~isempty(tok)
        tetrodes(end + 1) = str2double(tok{1}); %#ok<AGROW>
    end
end
tetrodes = unique(tetrodes);
end

function out = valid_files(d)
out = [];
for i = 1:numel(d)
    if d(i).isdir
        continue;
    end
    if strncmp(d(i).name, '._', 2)
        continue;
    end
    out = [out, d(i)]; %#ok<AGROW>
end
end

function nums = parse_number_list(txt)
nums = [];
if isempty(txt)
    return;
end
parts = regexp(txt, ',', 'split');
for i = 1:numel(parts)
    if isempty(parts{i})
        continue;
    end
    val = str2double(parts{i});
    if ~isnan(val)
        nums(end + 1) = val; %#ok<AGROW>
    end
end
nums = unique(nums);
end

function m = mean_no_nan(x)
x = x(~isnan(x));
if isempty(x)
    m = NaN;
else
    m = mean(x);
end
end

function tf = is_text_scalar(x)
tf = ischar(x) || (isstring(x) && isscalar(x));
end

function tf = is_text_or_cellstr(x)
if ischar(x) || (isstring(x) && isscalar(x))
    tf = true;
    return;
end
if iscell(x)
    tf = all(cellfun(@(c) ischar(c) || (isstring(c) && isscalar(c)), x));
    return;
end
tf = false;
end

function c = to_cellstr(x)
if isempty(x)
    c = {};
    return;
end
if ischar(x)
    c = {x};
    return;
end
if isstring(x)
    if isscalar(x)
        c = {char(x)};
    else
        c = cellstr(x(:));
    end
    return;
end
if iscell(x)
    c = cell(size(x));
    for i = 1:numel(x)
        if isstring(x{i})
            c{i} = char(x{i});
        else
            c{i} = x{i};
        end
    end
    return;
end
c = {};
end

function out = as_char(x)
if isstring(x)
    out = char(x);
else
    out = x;
end
end
