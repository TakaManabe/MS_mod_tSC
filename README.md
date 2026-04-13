# 『The medial septum controls hippocampal supra-theta oscillations』のための MATLAB code

## 概要

この repository には、theta-nested spectral components (tSCs)、single unit activity、optogenetic stimulus events の関係を解析するための MATLAB code が含まれています。single unit cluster と hippocampal tSCs は、Kilosort (https://github.com/MouseLand/Kilosort) と tSC extraction package (https://data.mrc.ox.ac.uk/data-set/tsc) を用いて抽出されています。

Király et al. による『The medial septum modulates hippocampal oscillations beyond the theta rhythm』の figure および supplementary figure は、以下 4 つの dataset に基づいて、MS_mod_tSC_main function で再現できます。
- freely moving mouse (hippocampal LFP - septal units)
- anesthetized rat (hippocampal LFP - septal units)
- anesthetized mouse (hippocampal LFP - septal units)
- mice における PV expressing septal neurons の optogenetic stimulation (hippocampal LFP, hippocampal units - septal stimulus events)

必要な data format と hierarchy を示した demo session data は、以下の link から利用できます。  
https://figshare.com/articles/dataset/Demo_data_for_the_MATLAB_code_for_analyzing_the_connection_between_the_medial_septum_and_hippocampal_oscillations_beyond_the_theta_rhythm_/22060964

(content: raw downsampled lfp、動物の position data (optional)、stimulation times (optional)、spike sorting と tSC extraction 後の preprocessed data)  
(例: demo session に対して `tSC_run_session_analyses ('L:\Balint\tsc\awake_mouse\','20161989','107108','10',1,1,'awake_mouse')`。input の詳細は code documentation を参照)

Session data は `tSC_run_session_analyses` function で preprocessing できます。preprocessed data は root directory の `Matrix.mat` と `ses_Matrix.mat` に保存されます。

## 内容

- data analysis 用の `.m` file
- license file

## インストール

`.m` file を MATLAB path に追加してください（数秒程度）。追加の installation は不要です。

## 依存関係

この code は、spike triggered average analysis のために hangya-matlab-code package の function を使用します。  
https://github.com/hangyabalazs/Hangya-Matlab-code

rhythmicity に基づく MS neuron の classification には、ms_sync_analysis package を使用しています。  
https://github.com/hangyabalazs/ms_sync_analysis

optogenetically stimulated hippocampal neuron の解析には Cellbase を使用しています。初回実行時は、まず Cellbase の initialization が必要です。  
https://github.com/hangyabalazs/CellBase

通常ではない data format の読み込みには、以下の function を使用しています。  
https://github.com/kwikteam/npy-matlab/blob/master/npy-matlab/readNPY.m  
https://github.com/open-ephys/analysis-tools/blob/master/load_open_ephys_data.m

## システム要件

- Windows 10 64bits
- Intel i7
- 32 GB RAM
- MatlabR2016a (Signal Processing Toolbox, Statistics and Machine Learning Toolbox, Wavelet Toolbox, Bioinformatics Toolbox)
- MatlabR2016a と MatlabR2018 でテスト済み
