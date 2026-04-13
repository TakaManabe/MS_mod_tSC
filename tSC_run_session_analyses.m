function tSC_run_session_analyses(mainpath, animal, session, ch, isstim, isrun, exp)
%TSC_RUN_SESSION_ANALYSES Analyze a session and update Matrix files.
%
%   This function can run in two modes:
%   1) Full mode: extracted tSC/theta files are available.
%   2) Raw-only mode: those files are missing; only analyses that do not
%      depend on extracted tSC files are executed.

% Resolve session data location and file prefix.
filename = [animal, session];
[fullpath, base] = resolve_session_fullpath(mainpath, animal, session, ch);

% Stimulation file is optional; if missing, run as non-stim.
stimFile1 = fullfile(mainpath, 'STIMULATIONS', [filename, '.mat']);
stimFile2 = fullfile(mainpath, 'STIMULATIONS', [animal, '_', session, '.mat']);
hasStimFile = exist(stimFile1, 'file') == 2 || exist(stimFile2, 'file') == 2;
isstim_eff = logical(isstim) && hasStimFile;
if isstim && ~isstim_eff
    warning('tSC_run_session_analyses:missingStimFile', ...
        'Missing stimulation file for %s (%s or %s). Running with isstim=0.', filename, stimFile1, stimFile2);
end

% Probe required extracted files.
f.theta_cycles = [fullpath, '.theta.cycles.', ch, '.mat'];
f.theta_cycleamp = [fullpath, '.theta.cycleamp.', ch, '.mat'];
f.theta_phase = [fullpath, '.theta.phase.', ch, '.mat'];
f.tsc_ica = [fullpath, '.tSCs.ica.', ch, '.mat'];
f.emd = [fullpath, '.emd.', ch, '.mat'];
f.emdif = [fullpath, '.emd.if.', ch, '.mat'];
f.eeg = [fullpath, '.eeg.', ch, '.mat'];

has_tsc_core = exist(f.theta_cycles, 'file') == 2 && exist(f.tsc_ica, 'file') == 2;
has_theta_amp = exist(f.theta_cycleamp, 'file') == 2;
has_theta_phase = exist(f.theta_phase, 'file') == 2;
has_emd = exist(f.emd, 'file') == 2 && exist(f.emdif, 'file') == 2;
has_eeg = exist(f.eeg, 'file') == 2;

% theta property and tSC presence correlations
if has_tsc_core && has_theta_amp
    tSC_thetaprop(mainpath, animal, session, ch, isstim_eff, isrun);
    tSC_ratio_oscstate(mainpath, animal, session, ch, isstim_eff, exp);
else
    warning('tSC_run_session_analyses:skipThetaProps', ...
        'Skipping tSC_thetaprop/tSC_ratio_oscstate for %s (missing extracted tSC/theta files).', filename);
end

% inter-spike-interval histograms (raw spikes only)
logISI(mainpath, animal, session, isstim_eff);

% Fourier and wavelet spectra (needs eeg)
if has_eeg
    neuron_spectra(mainpath, animal, session, ch, isstim_eff);
else
    warning('tSC_run_session_analyses:skipNeuronSpectra', ...
        'Skipping neuron_spectra for %s (missing %s).', filename, f.eeg);
end

% firing property and tSC presence correlations
if has_tsc_core && has_theta_phase
    tSC_neuron_firingprop(mainpath, animal, session, ch, isstim_eff);
else
    warning('tSC_run_session_analyses:skipNeuronFiringProp', ...
        'Skipping tSC_neuron_firingprop for %s (missing theta.phase/tSC files).', filename);
end

% tSC coupling of single units
if has_tsc_core && has_theta_phase && has_emd
    tSC_neuron_coupling(mainpath, animal, session, ch, isstim_eff, 1);
    tSC_neuron_coupling(mainpath, animal, session, ch, isstim_eff, 0);
else
    warning('tSC_run_session_analyses:skipNeuronCoupling', ...
        'Skipping tSC_neuron_coupling for %s (missing emd/theta.phase/tSC files).', filename);
end

end
