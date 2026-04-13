function [fullpath, base, dataBase] = resolve_session_fullpath(mainPath, animal, session, ch)
%RESOLVE_SESSION_FULLPATH Resolve session file prefix for multiple layouts.
%   [FULLPATH, BASE, DATABASE] = RESOLVE_SESSION_FULLPATH(MAINPATH, ANIMAL,
%   SESSION, CH) returns the prefix path used by session files such as
%   *.eeg.*, *.theta.cycles.*, *.tSCs.ica.*.
%
%   Supported layouts:
%   - <main>/<animal>/<session>/raw/<animal><session>_1.*
%   - <main>/<animal>/<session>/<animal><session>_1.*
%   - <main>/<animal>/<session>/raw/<animal>_<session>_1.*
%   - <main>/<animal>/<session>/<animal>_<session>_1.*
%
%   If no candidate matches, it falls back to the first *.eeg.* file found
%   in the data folder (prefix before '.eeg.<ch>.mat').

if isnumeric(ch)
    ch = num2str(ch);
elseif ~ischar(ch)
    ch = char(ch);
end

base = fullfile(mainPath, animal, session);
rawBase = fullfile(base, 'raw');
if isfolder(rawBase)
    dataBase = rawBase;
else
    dataBase = base;
end

filename1 = [animal, session];
filename2 = [animal, '_', session];

cand = {
    fullfile(dataBase, [filename1, '_1']), ...
    fullfile(dataBase, [filename2, '_1']), ...
    fullfile(dataBase, filename1), ...
    fullfile(dataBase, filename2) ...
    };

probeExt = {
    ['.eeg.', ch, '.mat'], ...
    ['.theta.cycles.', ch, '.mat'], ...
    ['.tSCs.ica.', ch, '.mat'], ...
    ['.theta.phase.', ch, '.mat'], ...
    ['.emd.', ch, '.mat'] ...
    };

for ci = 1:numel(cand)
    for ei = 1:numel(probeExt)
        if exist([cand{ci}, probeExt{ei}], 'file') == 2
            fullpath = cand{ci};
            return;
        end
    end
end

% Fallback: infer prefix from first eeg file in data folder.
eegFiles = dir(fullfile(dataBase, '*.eeg.*.mat'));
for i = 1:numel(eegFiles)
    nm = eegFiles(i).name;
    if strncmp(nm, '._', 2)
        continue;
    end
    tok = regexp(nm, '^(.*)\.eeg\.\d+\.mat$', 'tokens', 'once');
    if ~isempty(tok)
        fullpath = fullfile(dataBase, tok{1});
        return;
    end
end

% Final deterministic fallback.
fullpath = cand{1};
end
