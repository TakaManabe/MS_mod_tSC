function [animalTable, sessionTable] = print_stim_control_inventory(dataRoot, varargin)
%PRINT_STIM_CONTROL_INVENTORY Print and export STIM/control inventory.
%   [ANIMALTABLE, SESSIONTABLE] = PRINT_STIM_CONTROL_INVENTORY(DATAROOT)
%   scans STIM_Exp and awake_mouse_control datasets and summarizes:
%   - number of animals
%   - HC electrode channels (from *.eeg.*.mat)
%   - MS tetrode IDs (from TT*.mat)
%   - detected subregion tags from file names
%
%   Name-value options:
%       'StimRoot'     : full path to STIM dataset (default: DATAROOT/STIM_Exp)
%       'ControlRoot'  : full path to control dataset
%                        (default: DATAROOT/awake_mouse_control)
%       'OutputDir'    : output directory for CSV files (default: DATAROOT)
%       'Verbose'      : print summary to console (default: true)
%       'SaveTables'   : write CSV files (default: true)
%
%   Output files:
%       stim_control_session_inventory.csv
%       stim_control_animal_inventory.csv

ip = inputParser;
ip.addRequired('dataRoot', @is_text_scalar);
ip.addParameter('StimRoot', '', @is_text_or_cellstr);
ip.addParameter('ControlRoot', '', @is_text_or_cellstr);
ip.addParameter('OutputDir', '', @is_text_scalar);
ip.addParameter('Verbose', true, @islogical);
ip.addParameter('SaveTables', true, @islogical);
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

stimT = table();
for i = 1:numel(stimRoots)
    dsName = dataset_name_from_root(stimRoots{i}, 'STIM_Exp');
    ti = scan_dataset(stimRoots{i}, 'stim', dsName);
    stimT = concat_table(stimT, ti);
end

ctrlT = table();
for i = 1:numel(controlRoots)
    dsName = dataset_name_from_root(controlRoots{i}, 'awake_mouse_control');
    ti = scan_dataset(controlRoots{i}, 'control', dsName);
    ctrlT = concat_table(ctrlT, ti);
end

if isempty(stimT) && isempty(ctrlT)
    sessionTable = table();
elseif isempty(stimT)
    sessionTable = ctrlT;
elseif isempty(ctrlT)
    sessionTable = stimT;
else
    sessionTable = [stimT; ctrlT];
end

if isempty(sessionTable)
    warning('print_stim_control_inventory:NoData', ...
        'No sessions found in STIM/control roots.');
    animalTable = table();
    return;
end

animalTable = summarize_animals(sessionTable);

if opt.Verbose
    print_summary(animalTable, sessionTable, stimRoots, controlRoots);
end

if opt.SaveTables
    if ~isfolder(opt.OutputDir)
        mkdir(opt.OutputDir);
    end
    writetable(sessionTable, fullfile(opt.OutputDir, 'stim_control_session_inventory.csv'));
    writetable(animalTable, fullfile(opt.OutputDir, 'stim_control_animal_inventory.csv'));
end

end

function out = concat_table(a, b)
if isempty(a)
    out = b;
elseif isempty(b)
    out = a;
else
    out = [a; b];
end
end

function name = dataset_name_from_root(rootPath, fallback)
name = fallback;
if isempty(rootPath)
    return;
end
[~, n] = fileparts(rootPath);
if ~isempty(n)
    name = n;
end
end

function T = scan_dataset(rootPath, condition, datasetName)
rows = {};

if ~isfolder(rootPath)
    T = table();
    return;
end

animalDirs = list_subdirs(rootPath);
for ai = 1:numel(animalDirs)
    animal = animalDirs{ai};
    animalPath = fullfile(rootPath, animal);
    sessionBaseDirs = find_session_dirs_under_animal(animalPath);
    for si = 1:numel(sessionBaseDirs)
        baseDir = sessionBaseDirs{si};
        [dataDir, layout] = resolve_data_dir(baseDir);
        if isempty(dataDir)
            continue;
        end

        [~, sessionDirName] = fileparts(baseDir);
        session = parse_session_id(animal, sessionDirName);

        hcChannels = detect_hc_channels(dataDir);
        [ttFiles, msTetrodes] = detect_tt_files(baseDir, dataDir);
        [hcSubregions, msSubregions] = detect_subregions(baseDir, dataDir);

        rows(end + 1, :) = { ... %#ok<AGROW>
            datasetName, ...
            condition, ...
            rootPath, ...
            animal, ...
            session, ...
            baseDir, ...
            dataDir, ...
            layout, ...
            numel(hcChannels), ...
            number_list_to_text(hcChannels), ...
            numel(msTetrodes), ...
            number_list_to_text(msTetrodes), ...
            numel(ttFiles), ...
            text_list_to_text(hcSubregions), ...
            text_list_to_text(msSubregions)};
    end
end

if isempty(rows)
    T = table();
    return;
end

T = cell2table(rows, 'VariableNames', { ...
    'dataset', ...
    'condition', ...
    'root_path', ...
    'animal', ...
    'session', ...
    'base_dir', ...
    'data_dir', ...
    'layout', ...
    'n_hc_channels', ...
    'hc_channel_ids', ...
    'n_ms_tetrodes', ...
    'ms_tetrode_ids', ...
    'n_tt_units', ...
    'hc_subregions', ...
    'ms_subregions'});
end

function dirs = find_session_dirs_under_animal(animalPath)
dirs = {};
if ~isfolder(animalPath)
    return;
end

lvl1 = list_subdirs(animalPath);
for i = 1:numel(lvl1)
    p1 = fullfile(animalPath, lvl1{i});
    if is_session_dir(p1)
        dirs{end + 1} = p1; %#ok<AGROW>
        continue;
    end

    lvl2 = list_subdirs(p1);
    for j = 1:numel(lvl2)
        p2 = fullfile(p1, lvl2{j});
        if is_session_dir(p2)
            dirs{end + 1} = p2; %#ok<AGROW>
            continue;
        end

        lvl3 = list_subdirs(p2);
        for k = 1:numel(lvl3)
            p3 = fullfile(p2, lvl3{k});
            if is_session_dir(p3)
                dirs{end + 1} = p3; %#ok<AGROW>
            end
        end
    end
end

if isempty(dirs)
    return;
end
dirs = unique(dirs, 'stable');
end

function tf = is_session_dir(pathDir)
[dataDir, ~] = resolve_data_dir(pathDir);
tf = ~isempty(dataDir);
end

function animalTable = summarize_animals(sessionTable)
keys = unique(strcat(sessionTable.dataset, '|', sessionTable.animal));
rows = cell(numel(keys), 9);

for i = 1:numel(keys)
    parts = regexp(keys{i}, '\|', 'split');
    ds = parts{1};
    an = parts{2};
    idx = strcmp(sessionTable.dataset, ds) & strcmp(sessionTable.animal, an);
    sub = sessionTable(idx, :);

    hcAll = [];
    msAll = [];
    hcSubs = {};
    msSubs = {};
    for k = 1:height(sub)
        hcAll = unique([hcAll, parse_number_list(sub.hc_channel_ids{k})]); %#ok<AGROW>
        msAll = unique([msAll, parse_number_list(sub.ms_tetrode_ids{k})]); %#ok<AGROW>
        hcSubs = unique([hcSubs, parse_text_list(sub.hc_subregions{k})]); %#ok<AGROW>
        msSubs = unique([msSubs, parse_text_list(sub.ms_subregions{k})]); %#ok<AGROW>
    end

    rows(i, :) = { ...
        ds, ...
        an, ...
        height(sub), ...
        numel(hcAll), ...
        number_list_to_text(hcAll), ...
        numel(msAll), ...
        number_list_to_text(msAll), ...
        text_list_to_text(hcSubs), ...
        text_list_to_text(msSubs)};
end

animalTable = cell2table(rows, 'VariableNames', { ...
    'dataset', ...
    'animal', ...
    'n_sessions', ...
    'n_hc_channels', ...
    'hc_channel_ids', ...
    'n_ms_tetrodes', ...
    'ms_tetrode_ids', ...
    'hc_subregions', ...
    'ms_subregions'});
end

function print_summary(animalTable, sessionTable, stimRoots, controlRoots)
fprintf('=== STIM/control inventory ===\n');
fprintf('STIM roots   : %s\n', strjoin(stimRoots, ' ; '));
fprintf('Control roots: %s\n', strjoin(controlRoots, ' ; '));
fprintf('Sessions    : %d\n', height(sessionTable));
fprintf('Animals     : %d\n', height(animalTable));

datasets = unique(animalTable.dataset);
for i = 1:numel(datasets)
    ds = datasets{i};
    idx = strcmp(animalTable.dataset, ds);
    fprintf('\n[%s] animals=%d\n', ds, sum(idx));
    sub = animalTable(idx, :);
    for j = 1:height(sub)
        fprintf('  animal %s | sessions=%d | HC ch=%d (%s) | MS tetrodes=%d (%s) | HC subregion=%s | MS subregion=%s\n', ...
            sub.animal{j}, ...
            sub.n_sessions(j), ...
            sub.n_hc_channels(j), ...
            sub.hc_channel_ids{j}, ...
            sub.n_ms_tetrodes(j), ...
            sub.ms_tetrode_ids{j}, ...
            sub.hc_subregions{j}, ...
            sub.ms_subregions{j});
    end
end
end

function dirs = list_subdirs(parent)
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

function [dataDir, layout] = resolve_data_dir(baseDir)
dataDir = '';
layout = '';
if ~isfolder(baseDir)
    return;
end
rawDir = fullfile(baseDir, 'raw');
if isfolder(rawDir)
    dataDir = rawDir;
    layout = 'raw';
    return;
end

% Standard session_root format.
hasEeg = ~isempty(valid_files(dir(fullfile(baseDir, '*.eeg.*.mat'))));
hasTt = ~isempty(valid_files(dir(fullfile(baseDir, 'TT*.mat'))));
hasRadiatum = ~isempty(valid_files(dir(fullfile(baseDir, '*radiatum*.mat'))));
if hasEeg || hasTt || hasRadiatum
    dataDir = baseDir;
    layout = 'session_root';
end
end

function session = parse_session_id(animal, sessionDirName)
session = sessionDirName;
prefix = [animal, '_'];
if strncmp(sessionDirName, prefix, length(prefix))
    session = sessionDirName(length(prefix) + 1:end);
end
end

function hcChannels = detect_hc_channels(dataDir)
hcChannels = [];
eegFiles = valid_files(dir(fullfile(dataDir, '*.eeg.*.mat')));
for i = 1:numel(eegFiles)
    tok = regexp(eegFiles(i).name, '\.eeg\.(\d+)\.mat$', 'tokens', 'once');
    if ~isempty(tok)
        hcChannels(end + 1) = str2double(tok{1}); %#ok<AGROW>
    end
end
if isempty(hcChannels)
    regionPat = {'*radiatum*.mat', '*oriens*.mat', '*pyramid*.mat', '*ca1*.mat', '*ca3*.mat', '*dg*.mat'};
    for pi = 1:numel(regionPat)
        ff = valid_files(dir(fullfile(dataDir, regionPat{pi})));
        if ~isempty(ff)
            hcChannels = 1;
            break;
        end
    end
end
hcChannels = unique(hcChannels);
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

function [hcSubs, msSubs] = detect_subregions(baseDir, dataDir)
files = {};
f1 = valid_files(dir(fullfile(baseDir, '*.mat')));
f2 = valid_files(dir(fullfile(dataDir, '*.mat')));
for i = 1:numel(f1)
    files{end + 1} = lower(f1(i).name); %#ok<AGROW>
end
for i = 1:numel(f2)
    files{end + 1} = lower(f2(i).name); %#ok<AGROW>
end
files = unique(files);

hcTags = {'radiatum','oriens','pyramid','pyramidal','ca1','ca3','dg','dentate','hilus','subiculum'};
msTags = {'septum','ms','medialseptum'};

hcSubs = {};
for i = 1:numel(hcTags)
    tag = hcTags{i};
    if any(cellfun(@(x) ~isempty(strfind(x, tag)), files))
        hcSubs{end + 1} = tag; %#ok<AGROW>
    end
end

msSubs = {};
for i = 1:numel(msTags)
    tag = msTags{i};
    if any(cellfun(@(x) ~isempty(strfind(x, tag)), files))
        msSubs{end + 1} = tag; %#ok<AGROW>
    end
end

if isempty(hcSubs)
    hcSubs = {'unknown'};
end
if isempty(msSubs)
    msSubs = {'unknown'};
end
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

function txt = number_list_to_text(nums)
if isempty(nums)
    txt = '';
    return;
end
nums = unique(nums);
parts = cell(1, numel(nums));
for i = 1:numel(nums)
    parts{i} = sprintf('%d', nums(i));
end
txt = strjoin(parts, ',');
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

function txt = text_list_to_text(items)
if isempty(items)
    txt = 'unknown';
    return;
end
items = unique(items);
items = items(~cellfun(@isempty, items));
if isempty(items)
    txt = 'unknown';
else
    txt = strjoin(items, ',');
end
end

function items = parse_text_list(txt)
items = {};
if isempty(txt)
    return;
end
parts = regexp(txt, ',', 'split');
for i = 1:numel(parts)
    p = strtrim(parts{i});
    if isempty(p)
        continue;
    end
    if strcmpi(p, 'unknown')
        continue;
    end
    items{end + 1} = p; %#ok<AGROW>
end
items = unique(items);
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
