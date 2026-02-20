function MakeStimEvents2_tSC(sessionpath,ppch,varargin)
%MAKESTIMEVENTS2   Create stimulus events structure.
%   MAKESTIMEVENTS2_tSC(SESSIONPATH) constructs and saves stimulus events
%   structure ('StimEvents') for the stimulation session in the folder
%   SESSIONPATH. Stimulation pulses are detected as TTLs recorded by the
%   data acquisition system on channel PPCH and their number is verified
%   by the log file of the stimulation control software.

%   Modified from MakeStimEvents
%   Bálint Király
%   Institute of Experimental Medicine, Budapest, Hungary
%   kiraly.balint@koki.hu
%   24-Dec-2021

BurstSeparation = 0.5;

% Check session path
if ~isdir(sessionpath)
    error('Session path is not valid.');
end
cd(sessionpath)

% Load Open Ephys events
start=readNPY(fullfile(sessionpath,'timestamps.npy'));
timestamps1=double((readNPY(fullfile(sessionpath,'timestamps_stim.npy')))- start(1))/20000;
data1=double((readNPY(fullfile(sessionpath,'channels.npy'))));
info1.eventId=double((readNPY(fullfile(sessionpath,'channel_states.npy'))));
% PulsePal TTL onset and offsets
timestamps = timestamps1';
ppinx = data1==ppch;
pulses = timestamps(ppinx);   % PulsePal TTL's, both TTLon and TTLoff
eventIDs = info1.eventId(ppinx);
pulseon = pulses(eventIDs>0.5);
pulseoff = pulses(eventIDs<0.5);
nvalid_pulses = length(pulseon);  % calculating number of pulses

% Preallocate stimulus events
SE = struct;
SE.ProtocolName = cell(1,nvalid_pulses); % e.g. LaserStimProtocol2
SE.ProtocolID = cell(1,nvalid_pulses);   % 'S' for stim. and 'B' for behav. protocols
SE.ProtocolStart = nan(1,nvalid_pulses); % time stamp for protocol begin
SE.ProtocolEnd = nan(1,nvalid_pulses);   % time stamp for protocol end
SE.StimType = nan(1,nvalid_pulses);      % 1 for single pulse, 2 for burst stimualtion
SE.TrialStart = zeros(1,nvalid_pulses);  % should be 0

SE.BurstOn = nan(1,nvalid_pulses);       % onset of first pulse in the burst %csak az elejenel
SE.BurstOff = nan(1,nvalid_pulses);      % offset of last pulse in the burst
SE.BurstDur = nan(1,nvalid_pulses);      % duration of burst (between last pulse offset and first pulse onset
SE.BurstIBI = nan(1,nvalid_pulses);      % inter-burst interval
SE.BurstNPulse = nan(1,nvalid_pulses);   % number of pulses in burst (Events_EventStrings)
SE.PrevBurstOff = nan(1,nvalid_pulses);  % end of previous burst
SE.NextBurstOn = nan(1,nvalid_pulses);   % start of next burst
SE.PreBurstIBI = nan(1,nvalid_pulses);   % time since PrevBurstOff
SE.PostBurstIBI = nan(1,nvalid_pulses);  % time to NextBurstOn
SE.BurstID = nan(1,nvalid_pulses);       % burst rank in the protocol (NTrial)

SE.PulseOn = nan(1,nvalid_pulses);       % onset of stim. pulse
SE.PulseOff = nan(1,nvalid_pulses);      % offset of stim. pulse
SE.PulseDur = nan(1,nvalid_pulses);      % duration of light pulse (Events_EventStrings)
SE.PulseIPI = nan(1,nvalid_pulses);      % inter-pulse interval (Events_EventStrings)
SE.PulseFreq = nan(1,nvalid_pulses);     % pulse frequency (Events_EventStrings)
SE.PulsePower = nan(1,nvalid_pulses);    % stimulus intensity (Events_EventStrings)
SE.PrevPulseOff = nan(1,nvalid_pulses);  % offset of previous pulse
SE.NextPulseOn = nan(1,nvalid_pulses);   % onset of next pulse
SE.PrePulseIPI = nan(1,nvalid_pulses);   % time since PrevPulseOff
SE.PostPulseIPI = nan(1,nvalid_pulses);  % time to NextPulseOn

SE.FirstPulse = nan(1,nvalid_pulses);    % is this the first pulse in a burst?
SE.ZeroPulse = nan(1,nvalid_pulses);     % extrapolate one pulse time back
SE.LastPulse = nan(1,nvalid_pulses);     % is this the last pulse in a burst?
SE.PulseNum = nan(1,nvalid_pulses);      % pulse rank in the burst
SE.StimSerieNum = nan(1,nvalid_pulses);  % preak the pulsetrain to stim epochs
SE.BurstOn_tSC = nan(1,nvalid_pulses);

% Filling SE struct
SE.PulseOn = pulseon;
SE.PulseOff = pulseoff;

for currentEvent = 1:nvalid_pulses
    SE.PulseDur(currentEvent) = SE.PulseOff(currentEvent)-SE.PulseOn(currentEvent); % duration of stim. pulse (Events_EventStrings)
    SE.ProtocolName(currentEvent) = cellstr('Tagging'); % e.g. LaserStimProtocol2
    SE.ProtocolID(currentEvent) = cellstr('S');   % 'S' for stim. and 'B' for behav. protocols
    SE.ProtocolStart(currentEvent) = timestamps(3); % time stamp for protocol begin
    SE.ProtocolEnd(currentEvent) =  timestamps(end);   % time stamp for protocol end
    SE.StimType(currentEvent) = 2;      % 1 for single pulse, 2 for burst stimualtion
end

for currentEvent = 2:nvalid_pulses
    SE.PulseIPI(currentEvent) = SE.PulseOn(currentEvent)-SE.PulseOff(currentEvent-1); % inter-pulse interval (Events_EventStrings)
end

% BurstOn
burstinx = SE.PulseIPI>BurstSeparation;
SE.BurstOn(burstinx) = SE.PulseOn(burstinx);
SE.BurstOn(1) = SE.PulseOn(1);

% BurstOff
burstinx2 = find(burstinx) - 1;
SE.BurstOff(burstinx2) = SE.PulseOff(burstinx2);
SE.BurstOff(end) = SE.PulseOff(end);

% find separate burst epochs
burstinx_ext = [0,burstinx2,nvalid_pulses];
for i = 1:length(burstinx_ext)-1
    SE.StimSerieNum(burstinx_ext(i)+1:burstinx_ext(i+1)) = i;
end

% find tSC Burst starts
SE.BurstOn_tSC(abs(SE.PulseIPI-0.0778) < 0.001 & SE.StimSerieNum < max(SE.StimSerieNum)-3) = 1;
SE.BurstOn_tSC(abs(SE.PulseIPI-0.0661) < 0.001 & SE.StimSerieNum < max(SE.StimSerieNum)-3) = 2;
SE.BurstOn_tSC(abs(SE.PulseIPI-0.0678) < 0.001 & SE.StimSerieNum < max(SE.StimSerieNum)-3) = 3;
SE.BurstOn_tSC(abs(SE.PulseIPI-0.0733) < 0.001 & SE.StimSerieNum < max(SE.StimSerieNum)-3) = 4;

% Inter-burst interval
SE.PrevBurstOff = [SE.ProtocolStart(1) SE.BurstOff(1:end-1)];   % previous burst offset (first protocol start for first pulse)
SE.NextBurstOn = [SE.BurstOn(2:end) SE.ProtocolEnd(end)];   % next burst onset (last protocol end for last pulse)
SE.PreBurstIBI = SE.BurstOn - SE.PrevBurstOff;   % previous inter-burst interval
SE.PostBurstIBI = SE.NextBurstOn - SE.BurstOff;   % next inter-burst interval

% Inter-pulse interval
SE.PrevPulseOff = [SE.ProtocolStart(1) SE.PulseOff(1:end-1)];   % previous pulse offset (first protocol start for first pulse)
SE.NextPulseOn = [SE.PulseOn(2:end) SE.ProtocolEnd(end)];   % next pulse onset (last protocol end for last pulse)
SE.PrePulseIPI = SE.PulseOn - SE.PrevPulseOff;   % inter-pulse interval before pulse
SE.PostPulseIPI = SE.NextPulseOn - SE.PulseOff;   % inter-pulse interval after pulse

% Burst parameters
burston = find(~isnan(SE.BurstOn));
burstoff = find(~isnan(SE.BurstOff));
for currentEvent = 1:nvalid_pulses
    prevburston = burston(find(burston<=currentEvent,1,'last'));
    nextburstoff = burstoff(find(burstoff>=currentEvent,1,'first'));
    SE.BurstNPulse(currentEvent) = nextburstoff - prevburston + 1; % number of pulses in burst
    SE.BurstDur(currentEvent) = SE.PulseOn(nextburstoff) - SE.PulseOn(prevburston);
    SE.PulseFreq(currentEvent) = SE.BurstNPulse(currentEvent) / SE.BurstDur(currentEvent);
end

% Save
save(fullfile(sessionpath,'StimEvents.mat'),'-struct','SE'); %Save OE timestamps into a .mat file
