% Run inventory + directional table build on currently available datasets.
% Auto-detects dataset roots under dataRoot and classifies stim/control.

dataRoot = '/Volumes/T7_Taka/Minnesota/MSHC/Data_VargaV/23798184';
outDir = fullfile(dataRoot, 'analysis_directional');

fprintf('Data root: %s\n', dataRoot);

allDirs = list_immediate_subdirs(dataRoot);
allDirs = allDirs(~strcmpi(allDirs, 'Zip'));

if isempty(allDirs)
    error('No dataset folders found under %s', dataRoot);
end

datasetRoots = cellfun(@(d) fullfile(dataRoot, d), allDirs, 'UniformOutput', false);
isStim = false(size(datasetRoots));

for i = 1:numel(datasetRoots)
    isStim(i) = has_stim_evidence(datasetRoots{i});
end

stimRoots = datasetRoots(isStim);
controlRoots = datasetRoots(~isStim);

if isempty(controlRoots)
    controlRoots = datasetRoots;
end

fprintf('\nDetected dataset roots:\n');
for i = 1:numel(datasetRoots)
    tag = 'control';
    if isStim(i)
        tag = 'stim';
    end
    fprintf('  - [%s] %s\n', tag, datasetRoots{i});
end

if isempty(stimRoots)
    warning(['No stimulation evidence found (STIMULATIONS folder / StimEvents.mat / stim-like dataset name). ', ...
        'Proceeding with control-only sessions.']);
end

[animalTable, sessionTable] = print_stim_control_inventory(dataRoot, ...
    'StimRoot', stimRoots, ...
    'ControlRoot', controlRoots, ...
    'OutputDir', outDir, ...
    'Verbose', true, ...
    'SaveTables', true);

fprintf('\nInventory files:\n');
fprintf('  %s\n', fullfile(outDir, 'stim_control_animal_inventory.csv'));
fprintf('  %s\n', fullfile(outDir, 'stim_control_session_inventory.csv'));
fprintf('Rows: animals=%d, sessions=%d\n', height(animalTable), height(sessionTable));

[eventTable, ~] = build_directional_table_stim_control(dataRoot, ...
    'StimRoot', stimRoots, ...
    'ControlRoot', controlRoots, ...
    'OutputDir', outDir, ...
    'SamplingRate', 1000, ...
    'WindowMs', 200, ...
    'MinInterEventMs', 300, ...
    'RngSeed', 42, ...
    'Verbose', true);

fprintf('\nDirectional table rows: %d\n', height(eventTable));
fprintf('Directional outputs:\n');
fprintf('  %s\n', fullfile(outDir, 'directional_event_table_stim_control.csv'));
fprintf('  %s\n', fullfile(outDir, 'directional_event_table_stim_control.mat'));

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
    return;
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
