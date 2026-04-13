function results = run_full_pipeline_externalprep_and_matlab(dataRoot, varargin)
%RUN_FULL_PIPELINE_EXTERNALPREP_AND_MATLAB
% Step1 + Step2 unified loop:
%   Step1: external preprocessing (Kilosort / tSC extraction) command(s)
%   Step2: tSC_run_session_analyses
%
% This runner is intentionally generic because this repository does not
% include the external tSC extraction implementation itself.
%
% Example:
%   results = run_full_pipeline_externalprep_and_matlab( ...
%       '/Volumes/T7_Taka/Minnesota/MSHC/Data_VargaV/23798184', ...
%       'KilosortCommandTemplate', '', ...
%       'TscCommandTemplate', 'python /path/to/tsc_extract.py --session {SESSION_ROOT_Q} --ch {CH}');
%
% Supported placeholders in command templates:
%   {DATA_ROOT}, {MAINPATH}, {SESSION_ROOT}, {DATA_DIR}, {ANIMAL},
%   {SESSION}, {CH}, {FILENAME}
%   and quoted variants:
%   {DATA_ROOT_Q}, {MAINPATH_Q}, {SESSION_ROOT_Q}, {DATA_DIR_Q}

if nargin < 1 || isempty(dataRoot)
    dataRoot = '/Volumes/T7_Taka/Minnesota/MSHC/Data_VargaV/23798184';
end

ip = inputParser;
ip.addRequired('dataRoot', @is_text_scalar);
ip.addParameter('KilosortCommandTemplate', '', @is_text_scalar);
ip.addParameter('TscCommandTemplate', '', @is_text_scalar);
ip.addParameter('ForcePrep', false, @islogical);
ip.addParameter('DryRun', false, @islogical);
ip.addParameter('StopOnError', false, @islogical);
ip.parse(dataRoot, varargin{:});
opt = ip.Results;

dataRoot = as_char(dataRoot);
opt.KilosortCommandTemplate = as_char(opt.KilosortCommandTemplate);
opt.TscCommandTemplate = as_char(opt.TscCommandTemplate);

datasets = { ...
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
    'prep_status', {}, ...
    'analysis_status', {}, ...
    'status', {}, ...
    'message', {});

fprintf('Data root: %s\n', dataRoot);
if isempty(opt.KilosortCommandTemplate) && isempty(opt.TscCommandTemplate)
    fprintf('[WARN] No external prep command templates set. Only already-preprocessed sessions can run.\n');
end

for di = 1:numel(datasets)
    ds = datasets{di};
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
                'prep_status', 'skip', ...
                'analysis_status', 'skip', ...
                'status', 'skip', ...
                'message', '');

            if isempty(dataDir)
                row.message = 'missing session folder';
                results(end + 1) = row; %#ok<AGROW>
                fprintf('[SKIP] %s/%s : %s\n', animal, session, row.message);
                continue;
            end

            ch = detect_channel_for_session(dataDir, sessionRoot);
            if isempty(ch)
                row.message = 'cannot detect channel from eeg/tSC files';
                results(end + 1) = row; %#ok<AGROW>
                fprintf('[SKIP] %s/%s : %s\n', animal, session, row.message);
                continue;
            end
            row.ch = ch;

            fname1 = [animal, session, '.mat'];
            fname2 = [animal, '_', session, '.mat'];
            stim1 = fullfile(mainpath, 'STIMULATIONS', fname1);
            stim2 = fullfile(mainpath, 'STIMULATIONS', fname2);
            isstim = exist(stim1, 'file') == 2 || exist(stim2, 'file') == 2;
            row.isstim = isstim;

            [ready, missingList] = has_required_extracted_files(dataDir, sessionRoot, ch);
            if ready && ~opt.ForcePrep
                row.prep_status = 'already_ready';
                fprintf('[PREP] %s/%s ch=%s : already ready\n', animal, session, ch);
            else
                [prepOk, prepMsg] = run_external_prep_commands( ...
                    dataRoot, mainpath, sessionRoot, dataDir, animal, session, ch, ...
                    opt.KilosortCommandTemplate, opt.TscCommandTemplate, opt.DryRun);
                if prepOk
                    row.prep_status = 'ok';
                    if ~isempty(prepMsg)
                        fprintf('%s\n', prepMsg);
                    end
                else
                    row.prep_status = 'error';
                    row.message = prepMsg;
                    row.status = 'error';
                    results(end + 1) = row; %#ok<AGROW>
                    fprintf('[ERR ] %s/%s PREP: %s\n', animal, session, prepMsg);
                    if opt.StopOnError
                        error('Stopping on prep error.');
                    end
                    continue;
                end

                [ready, missingList] = has_required_extracted_files(dataDir, sessionRoot, ch);
            end

            if ~ready
                row.prep_status = 'incomplete';
                row.analysis_status = 'skip';
                row.status = 'skip';
                row.message = ['missing extracted files: ', strjoin(missingList, ', ')];
                results(end + 1) = row; %#ok<AGROW>
                fprintf('[SKIP] %s/%s : %s\n', animal, session, row.message);
                continue;
            end

            fprintf('[RUN ] %s/%s ch=%s isstim=%d isrun=%d exp=%s\n', ...
                animal, session, ch, isstim, ds.isrun, ds.exp);
            try
                tSC_run_session_analyses(mainpath, animal, session, ch, isstim, ds.isrun, ds.exp);
                row.analysis_status = 'ok';
                row.status = 'ok';
                row.message = '';
                fprintf('[ OK ] %s/%s\n', animal, session);
            catch ME
                row.analysis_status = 'error';
                row.status = 'error';
                row.message = ME.message;
                fprintf('[ERR ] %s/%s ANALYSIS: %s\n', animal, session, ME.message);
                if opt.StopOnError
                    error('Stopping on analysis error.');
                end
            end

            results(end + 1) = row; %#ok<AGROW>
        end
    end
end

print_summary(results);

end

function [ready, missingList] = has_required_extracted_files(dataDir, sessionRoot, ch)
required = { ...
    ['*.tSCs.ica.', ch, '.mat'], ...
    ['*.theta.cycles.', ch, '.mat'], ...
    ['*.theta.phase.', ch, '.mat'], ...
    ['*.theta.cycleamp.', ch, '.mat'], ...
    ['*.emd.', ch, '.mat'], ...
    ['*.emd.if.', ch, '.mat'] ...
    };

missingList = {};
for i = 1:numel(required)
    p = required{i};
    ok = ~isempty(valid_files(dir(fullfile(dataDir, p)))) || ...
         ~isempty(valid_files(dir(fullfile(sessionRoot, p))));
    if ~ok
        missingList{end + 1} = p; %#ok<AGROW>
    end
end
ready = isempty(missingList);
end

function [ok, msg] = run_external_prep_commands( ...
    dataRoot, mainpath, sessionRoot, dataDir, animal, session, ch, kTemplate, tTemplate, dryRun)

ok = true;
msg = '';

commands = {};
if ~isempty(kTemplate)
    commands{end + 1} = kTemplate; %#ok<AGROW>
end
if ~isempty(tTemplate)
    commands{end + 1} = tTemplate; %#ok<AGROW>
end

if isempty(commands)
    ok = false;
    msg = 'external prep command templates are empty';
    return;
end

filename = [animal, session];
kv = struct( ...
    'DATA_ROOT', dataRoot, ...
    'MAINPATH', mainpath, ...
    'SESSION_ROOT', sessionRoot, ...
    'DATA_DIR', dataDir, ...
    'ANIMAL', animal, ...
    'SESSION', session, ...
    'CH', ch, ...
    'FILENAME', filename, ...
    'DATA_ROOT_Q', shq(dataRoot), ...
    'MAINPATH_Q', shq(mainpath), ...
    'SESSION_ROOT_Q', shq(sessionRoot), ...
    'DATA_DIR_Q', shq(dataDir));

for i = 1:numel(commands)
    cmd = fill_template(commands{i}, kv);
    if dryRun
        fprintf('[DRY ] %s\n', cmd);
        continue;
    end
    fprintf('[PREP] %s\n', cmd);
    [st, out] = system(cmd);
    if st ~= 0
        ok = false;
        msg = sprintf('prep command failed (status %d): %s', st, short_text(out, 600));
        return;
    end
end
end

function out = fill_template(template, kv)
out = template;
names = fieldnames(kv);
for i = 1:numel(names)
    key = names{i};
    out = strrep(out, ['{', key, '}'], kv.(key));
end
end

function txt = short_text(s, n)
if nargin < 2
    n = 600;
end
s = strtrim(s);
if numel(s) <= n
    txt = s;
else
    txt = [s(1:n), ' ...'];
end
end

function q = shq(s)
% Single-quote for shell command.
q = ['''', strrep(s, '''', '''"''"'''), ''''];
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

function ch = detect_channel_for_session(dataDir, sessionRoot)
ch = detect_channel_from_pattern(dataDir, '\.tSCs\.ica\.(\d+)\.mat$');
if ~isempty(ch)
    return;
end
ch = detect_channel_from_pattern(dataDir, '\.eeg\.(\d+)\.mat$');
if ~isempty(ch)
    return;
end
ch = detect_channel_from_pattern(sessionRoot, '\.eeg\.(\d+)\.mat$');
end

function ch = detect_channel_from_pattern(folderPath, rx)
ch = '';
if ~isfolder(folderPath)
    return;
end
files = valid_files(dir(fullfile(folderPath, '*.mat')));
vals = [];
for i = 1:numel(files)
    tok = regexp(files(i).name, rx, 'tokens', 'once');
    if ~isempty(tok)
        vals(end + 1) = str2double(tok{1}); %#ok<AGROW>
    end
end
if isempty(vals)
    return;
end
vals = sort(unique(vals));
ch = sprintf('%d', vals(1));
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
prepSt = {results.prep_status};
anaSt = {results.analysis_status};
fprintf('\n=== Summary ===\n');
fprintf('Total sessions: %d\n', numel(results));
fprintf('Final OK      : %d\n', sum(strcmp(st, 'ok')));
fprintf('Final ERROR   : %d\n', sum(strcmp(st, 'error')));
fprintf('Final SKIP    : %d\n', sum(strcmp(st, 'skip')));
fprintf('Prep OK       : %d\n', sum(strcmp(prepSt, 'ok') | strcmp(prepSt, 'already_ready')));
fprintf('Prep ERROR    : %d\n', sum(strcmp(prepSt, 'error')));
fprintf('Analysis OK   : %d\n', sum(strcmp(anaSt, 'ok')));
fprintf('Analysis ERROR: %d\n', sum(strcmp(anaSt, 'error')));
end

function tf = is_text_scalar(x)
tf = ischar(x) || (isstring(x) && isscalar(x));
end

function out = as_char(x)
if isstring(x)
    out = char(x);
else
    out = x;
end
end

