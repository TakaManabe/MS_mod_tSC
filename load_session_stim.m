function stim = load_session_stim(mainPath, animal, session)
%LOAD_SESSION_STIM Load stimulation vector with multiple naming schemes.
%   STIM = LOAD_SESSION_STIM(MAINPATH, ANIMAL, SESSION) loads the
%   stimulation vector from MAINPATH/STIMULATIONS.

stimDir = fullfile(mainPath, 'STIMULATIONS');
fname1 = [animal, session];
fname2 = [animal, '_', session];

candidates = {
    fullfile(stimDir, [fname1, '.mat']), ...
    fullfile(stimDir, [fname2, '.mat']) ...
    };

for i = 1:numel(candidates)
    if exist(candidates{i}, 'file') ~= 2
        continue;
    end

    s = load(candidates{i}, 'stim');
    if isfield(s, 'stim')
        stim = s.stim;
    else
        s2 = load(candidates{i});
        fn = fieldnames(s2);
        if isempty(fn)
            error('load_session_stim:EmptyFile', ...
                'No variables found in stimulation file: %s', candidates{i});
        end
        stim = s2.(fn{1});
    end

    if iscell(stim)
        stim = cell2mat(stim);
    end
    return;
end

error('load_session_stim:MissingStimFile', ...
    'Missing stimulation file for %s/%s under %s', animal, session, stimDir);
end
