function results = batch_preprocess_all_23798184(dataRoot)
%BATCH_PREPROCESS_ALL_23798184 Run tSC_run_session_analyses for all sessions.
%   RESULTS = BATCH_PREPROCESS_ALL_23798184(DATAROOT) scans known dataset
%   folders under DATAROOT and runs tSC_run_session_analyses on every
%   animal/session pair that has the required files.
%
%   If DATAROOT is empty, default path is used:
%   /Volumes/T7_Taka/Minnesota/MSHC/Data_VargaV/23798184

if nargin < 1 || isempty(dataRoot)
    dataRoot = '/Volumes/T7_Taka/Minnesota/MSHC/Data_VargaV/23798184';
end

datasets = {
    struct('folder','awake_mouse','exp','awake_mouse','isrun',1), ...
    struct('folder','anast_rat_data_msmodtSC','exp','anesthetized_rat','isrun',0), ...
    struct('folder','anast_mouse_data_msmodtSC','exp','anesthetized_mouse','isrun',0) ...
    };

results = struct( ...
    'dataset', {}, ...
    'animal', {}, ...
    'session', {}, ...
    'layout', {}, ...
    'ch', {}, ...
    'isstim', {}, ...
    'status', {}, ...
    'message', {});

fprintf('Data root: %s\n', dataRoot);

for di = 1:numel(datasets)
    cfg = datasets{di};
    mainpath = fullfile(dataRoot, cfg.folder);
    if ~isfolder(mainpath)
        fprintf('[SKIP DATASET] %s (missing)\n', mainpath);
        continue;
    end

    animals = list_numeric_subdirs(mainpath);
    fprintf('\n=== Dataset: %s (animals: %d) ===\n', cfg.folder, numel(animals));

    for ai = 1:numel(animals)
        animal = animals{ai};
        animalPath = fullfile(mainpath, animal);
        sessions = list_numeric_subdirs(animalPath);

        for si = 1:numel(sessions)
            session = sessions{si};
            base = fullfile(mainpath, animal, session);
            filename = [animal, session];
            [dataPath, dataLayout] = resolve_data_path(base);

            entry = struct( ...
                'dataset', cfg.folder, ...
                'animal', animal, ...
                'session', session, ...
                'layout', dataLayout, ...
                'ch', '', ...
                'isstim', 0, ...
                'status', 'skip', ...
                'message', '');

            if isempty(dataPath)
                entry.message = 'missing session folder';
                results(end + 1) = entry; %#ok<AGROW>
                fprintf('[SKIP] %s/%s : %s\n', animal, session, entry.message);
                continue;
            end

            ch = detect_channel(dataPath);
            if isempty(ch)
                entry.message = ['missing channel files (*.eeg.* / *.tSCs.ica.* / *.theta.cycles.*) in ', dataLayout];
                results(end + 1) = entry; %#ok<AGROW>
                fprintf('[SKIP] %s/%s : %s\n', animal, session, entry.message);
                continue;
            end

            stimFile1 = fullfile(mainpath, 'STIMULATIONS', [filename, '.mat']);
            stimFile2 = fullfile(mainpath, 'STIMULATIONS', [animal, '_', session, '.mat']);
            isstim = exist(stimFile1, 'file') == 2 || exist(stimFile2, 'file') == 2;

            entry.ch = ch;
            entry.isstim = isstim;
            fprintf('[RUN ] %s/%s layout=%s ch=%s isstim=%d isrun=%d exp=%s\n', ...
                animal, session, dataLayout, ch, isstim, cfg.isrun, cfg.exp);

            try
                tSC_run_session_analyses(mainpath, animal, session, ch, isstim, cfg.isrun, cfg.exp);
                entry.status = 'ok';
                entry.message = '';
                fprintf('[ OK ] %s/%s\n', animal, session);
            catch ME
                entry.status = 'error';
                entry.message = ME.message;
                fprintf('[ERR ] %s/%s : %s\n', animal, session, ME.message);
            end

            results(end + 1) = entry; %#ok<AGROW>
        end
    end
end

summarize_results(results);
write_log_file(dataRoot, results);

end

function names = list_numeric_subdirs(parentPath)
d = dir(parentPath);
d = d([d.isdir]);
names = {};
for i = 1:numel(d)
    nm = d(i).name;
    if strcmp(nm, '.') || strcmp(nm, '..')
        continue;
    end
    if strncmp(nm, '._', 2)
        continue;
    end
    if isempty(regexp(nm, '^\d+$', 'once'))
        continue;
    end
    names{end + 1} = nm; %#ok<AGROW>
end
end

function [dataPath, dataLayout] = resolve_data_path(base)
rawPath = fullfile(base, 'raw');
if isfolder(rawPath)
    dataPath = rawPath;
    dataLayout = 'raw';
elseif isfolder(base)
    dataPath = base;
    dataLayout = 'session_root';
else
    dataPath = '';
    dataLayout = 'missing';
end
end

function ch = detect_channel(dataPath)
% Prefer channel from extracted tSC outputs.
ch = '';

ch = find_channel_from_pattern(dataPath, '*.tSCs.ica.*.mat', '\.tSCs\.ica\.(\d+)\.mat$');
if ~isempty(ch)
    return;
end

ch = find_channel_from_pattern(dataPath, '*.theta.cycles.*.mat', '\.theta\.cycles\.(\d+)\.mat$');
if ~isempty(ch)
    return;
end

% Fallback to eeg channel for raw-only preprocessing mode.
ch = find_channel_from_pattern(dataPath, '*.eeg.*.mat', '\.eeg\.(\d+)\.mat$');
if ~isempty(ch)
    return;
end

% Last fallback: if spikes exist, run raw-only parts with a dummy channel.
tt = dir(fullfile(dataPath, 'TT*.mat'));
if ~isempty(tt)
    ch = '1';
    return;
end
end

function ch = find_channel_from_pattern(dataPath, filePattern, regexPattern)
ch = '';
files = dir(fullfile(dataPath, filePattern));
if isempty(files)
    return;
end

vals = cell(1, numel(files));
for i = 1:numel(files)
    tok = regexp(files(i).name, regexPattern, 'tokens', 'once');
    if ~isempty(tok)
        vals{i} = tok{1};
    end
end
vals = vals(~cellfun('isempty', vals));
if isempty(vals)
    return;
end

% Deterministic choice in case multiple channels are found.
nums = cellfun(@str2double, vals);
[~, idx] = sort(nums);
vals = vals(idx);
ch = vals{1};
end

function summarize_results(results)
if isempty(results)
    fprintf('\nNo sessions found.\n');
    return;
end

st = {results.status};
nOk = sum(strcmp(st, 'ok'));
nErr = sum(strcmp(st, 'error'));
nSkip = sum(strcmp(st, 'skip'));

fprintf('\n=== Summary ===\n');
fprintf('Total: %d\n', numel(results));
fprintf('OK   : %d\n', nOk);
fprintf('ERROR: %d\n', nErr);
fprintf('SKIP : %d\n', nSkip);
end

function write_log_file(dataRoot, results)
if isempty(results)
    return;
end

ts = datestr(now, 'yyyymmdd_HHMMSS');
logPath = fullfile(dataRoot, ['batch_preprocess_log_', ts, '.txt']);
fid = fopen(logPath, 'w');
if fid < 0
    fprintf('Could not write log file: %s\n', logPath);
    return;
end

    fprintf(fid, 'dataset\tanimal\tsession\tlayout\tch\tisstim\tstatus\tmessage\n');
for i = 1:numel(results)
    r = results(i);
    msg = strrep(r.message, sprintf('\n'), ' ');
        fprintf(fid, '%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\n', ...
            r.dataset, r.animal, r.session, r.layout, r.ch, r.isstim, r.status, msg);
end
fclose(fid);
fprintf('Log file: %s\n', logPath);
end
