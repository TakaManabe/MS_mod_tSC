function results = run_all_sessions_forloop_original(dataRoot)
%RUN_ALL_SESSIONS_FORLOOP_ORIGINAL
% Minimal for-loop runner for original tSC preprocessing pipeline.
%
% This uses the existing entry point:
%   tSC_run_session_analyses(mainpath, animal, session, ch, isstim, isrun, exp)
%
% Required preconditions per session:
%   - tSC extraction outputs exist (*.tSCs.ica.<ch>.mat etc.)
%   - clustered spikes exist (TT*.mat)
%
% Example:
%   results = run_all_sessions_forloop_original('/Volumes/T7_Taka/Minnesota/MSHC/Data_VargaV/23798184');

if nargin < 1 || isempty(dataRoot)
    dataRoot = '/Volumes/T7_Taka/Minnesota/MSHC/Data_VargaV/23798184';
end

cfg = { ...
    struct('folder', 'awake_mouse', 'exp', 'awake_mouse', 'isrun', 1), ...
    struct('folder', 'anast_rat_data_msmodtSC', 'exp', 'anesthetized_rat', 'isrun', 0), ...
    struct('folder', 'anast_mouse_data_msmodtSC', 'exp', 'anesthetized_mouse', 'isrun', 0) ...
    };

results = struct( ...
    'dataset', {}, ...
    'animal', {}, ...
    'session', {}, ...
    'ch', {}, ...
    'isstim', {}, ...
    'status', {}, ...
    'message', {});

fprintf('Data root: %s\n', dataRoot);

for di = 1:numel(cfg)
    ds = cfg{di};
    mainpath = fullfile(dataRoot, ds.folder);
    if ~isfolder(mainpath)
        fprintf('[SKIP DATASET] %s (missing)\n', mainpath);
        continue;
    end

    animals = list_numeric_subdirs(mainpath);
    fprintf('\n=== Dataset: %s (animals: %d) ===\n', ds.folder, numel(animals));

    for ai = 1:numel(animals)
        animal = animals{ai};
        sessions = list_numeric_subdirs(fullfile(mainpath, animal));

        for si = 1:numel(sessions)
            session = sessions{si};
            sessionRoot = fullfile(mainpath, animal, session);
            dataDir = pick_data_dir(sessionRoot);

            row = struct( ...
                'dataset', ds.folder, ...
                'animal', animal, ...
                'session', session, ...
                'ch', '', ...
                'isstim', 0, ...
                'status', 'skip', ...
                'message', '');

            if isempty(dataDir)
                row.message = 'missing session folder';
                results(end + 1) = row; %#ok<AGROW>
                fprintf('[SKIP] %s/%s : %s\n', animal, session, row.message);
                continue;
            end

            ch = detect_tsc_channel(dataDir);
            if isempty(ch)
                row.message = 'missing *.tSCs.ica.<ch>.mat';
                results(end + 1) = row; %#ok<AGROW>
                fprintf('[SKIP] %s/%s : %s\n', animal, session, row.message);
                continue;
            end

            fname1 = [animal, session, '.mat'];
            fname2 = [animal, '_', session, '.mat'];
            stim1 = fullfile(mainpath, 'STIMULATIONS', fname1);
            stim2 = fullfile(mainpath, 'STIMULATIONS', fname2);
            isstim = exist(stim1, 'file') == 2 || exist(stim2, 'file') == 2;

            row.ch = ch;
            row.isstim = isstim;

            fprintf('[RUN ] %s/%s ch=%s isstim=%d isrun=%d exp=%s\n', ...
                animal, session, ch, isstim, ds.isrun, ds.exp);

            try
                tSC_run_session_analyses(mainpath, animal, session, ch, isstim, ds.isrun, ds.exp);
                row.status = 'ok';
                fprintf('[ OK ] %s/%s\n', animal, session);
            catch ME
                row.status = 'error';
                row.message = ME.message;
                fprintf('[ERR ] %s/%s : %s\n', animal, session, ME.message);
            end

            results(end + 1) = row; %#ok<AGROW>
        end
    end
end

print_summary(results);

end

function dataDir = pick_data_dir(sessionRoot)
if ~isfolder(sessionRoot)
    dataDir = '';
    return;
end
rawDir = fullfile(sessionRoot, 'raw');
if isfolder(rawDir)
    dataDir = rawDir;
else
    dataDir = sessionRoot;
end
end

function ch = detect_tsc_channel(dataDir)
ch = '';
files = dir(fullfile(dataDir, '*.tSCs.ica.*.mat'));
if isempty(files)
    return;
end

vals = nan(1, numel(files));
for i = 1:numel(files)
    tok = regexp(files(i).name, '\.tSCs\.ica\.(\d+)\.mat$', 'tokens', 'once');
    if ~isempty(tok)
        vals(i) = str2double(tok{1});
    end
end
vals = vals(~isnan(vals));
if isempty(vals)
    return;
end
vals = sort(vals);
ch = sprintf('%d', vals(1));
end

function names = list_numeric_subdirs(parentPath)
if ~isfolder(parentPath)
    names = {};
    return;
end
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
names = sort(names);
end

function print_summary(results)
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

