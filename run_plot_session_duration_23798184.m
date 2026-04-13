% Quick runner: session-length inventory + distribution plots.
dataRoot = '/Volumes/T7_Taka/Minnesota/MSHC/Data_VargaV/23798184';

sessionLenTable = plot_session_duration_23798184(dataRoot, ...
    'SamplingRate', 1000, ...
    'Verbose', true);

fprintf('\nRows in sessionLenTable: %d\n', height(sessionLenTable));
