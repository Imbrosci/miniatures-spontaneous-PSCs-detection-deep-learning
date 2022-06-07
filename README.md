# Spontaneous and miniatures postsynaptic currents (PSCs) detection

### This repository provides an algorithm, primarily based on deep learning, to automate the detection of miniatures and spontaneous inward postsynaptic currents (PSCs) in neurophysiological patch-clamp recordings.

Here below an example of the performance of the algorithm on a voltage clamp recording. The blue dots represents the spontaneous PSCs detected by the algorithm. 

![Alt text](/example.jpg?raw=true)

The model is currently under revision. Once the revision will be finished, a detailed explanation on how to use it will follow. 
Last update: 05.06.22

## Prerequisites
Anaconda or miniconda and python 3. We recommend to use python 3.10 and to create a virtual environment as follows: conda create --name my_env python=3.10.

## Installation
Clone the repository on your local machine with the following command: git clone https://github.com/Imbrosci/spontaneous-postsynaptic-currents-detection.git. 
Navigate to the cloned repository, spontaneous-postsynaptic-currents-detection and:
1) install the required libraries with: pip install -r requirements.txt;
2) install tensorflow with: conda install -c conda-forge tensorflow;
3) create two folders, named recordings and results, with: mkdir recordings results.

## Preliminary step before starting the analysis 
Before starting the analysis, there are two simple, but important steps to do.
1) move the recordings to be analysed in the folder recordings. The accepted formats are abf and txt files.
2) fill the metadata.xlsx file. A template should be found in the folder metadata.
Each column of the metadata.xlsx file should be filled as explained below:
- Name of recording: name of the file with the recording including the extension (.abf or .txt). The files listed should be found in the recording folder. 
- Channels to use: if your recording file contains only one channel, enter 1; in case your recording file contains more channels, you should indicate which channels should be analysed. If you want to analyse more channels of a file you can enter the name of the channels separated by a comma or you could keep one row per channel. The algorithm will number the channels found in the recording file in ascending order starting from 1.
- Sampling rate (Hz): sampling rate in Herz.
- Analysis start at sweep (number): sweep from which the analysis should start. If your recordings do not have multiple sweeps, select 1. If the recordings have multiple sweeps, the sweeps should have equal length.
- Cut sweeps first part (ms): this option offers the possibility to cut out, from the mPSCs / sPSCs analysis, a portion of trace at the beginning of each sweep. This may be relevant if a test pulse (to check, for instance, the stability of a patch clamp recording), or another kind of stimulation, was delivered at the beginning of each sweep. For example, to remove the first half a second of trace from the beginning of each sweep, enter 500.
- Cut sweeps last part (ms): similarly, to the option above, this option offers the possibility to cut out a portion of trace from the end of each sweep backward. For example, to remove the last half a second of trace at the end of each sweep, enter 500.
- Analysis length (min): enter how many minutes of recording should be analysed. Be aware that analysing many minutes at once (especially if a recording was acquired with a high sampling rate) may cause difficulties in displaying or proof-reading the results. If the recording is shorter than the time indicated in this option, the algorithm will just analyse the recordings until the end. 

## Starting the analysis
To start the recording, navigate in the repository folder and type: python running_analysis.py. If everything is running correctly, after a few moments, some details about the analysis will be printed.
If there is already a results.xlsx file produced by a former analysis the program will ask you the permission to overwrite the results. To do not overwrite the previous results, just change the name of the results.xlsx file or move it in another folder. 

## Excel file containing the results
Once the results are finished, a results.xlsx file will be generated and can be found under the folder results. 
In the results.xlsx file, there will be a sheet for each analysed recording_channel and summary results sheet.
The sheets for single recording_channel report the x and y position (in ms and pA, respectively), the interevent interval (between event + 1 and event) (in ms) and the amplitude (in pA) for each detected event.
The summary results sheet contain the following information for each recording:
- Recording filename;
- Channel;
- Sampling rate (Hz);
- Analysis start at sweep (number);
- Cut sweeps first part (ms);
- Cut sweeps last part (ms);
- Analysis length (sec) (this may be shorter than specified in the metadata if the recording was shorter);
- Average interevent interval (ms);
- Average amplitude (pA);
- Average 10-90% rise time (ms) (it is calculated from the mean signal of all detected event);
- Average 90-10% decay time (ms) (it is calculated from the mean signal of all detected event);
- Stdev of the baseline signal (pA) (Stdev = standard deviation, it is calculated from the trace after excluding the detected events - - it may provide an information about the quality of the recording and it can be used to exclude certain recordings from the analysis, if, for example, the average amplitude is not larger than 3 times the Stdev);
- Manually revised (if the results were manually corrected or not, see proof-read results section for details).

## Check results
To check the quality of the results use the command: python display_results.py. The program will ask to enter the file name (including extension) and the channel to display.
Two windows will appear. The first window displays the recorded signal and the detected events (as blue dots), the second window represents the average signal from all detected events. The rise (10-90%) and the decay (90-10%) time will be also displayed as cyan and pink dots, respectively. 

## Proof-read results
If the results are not satisfactory, it is also possible to correct them. 
To correct the results (add false negatives or delete false positives), use the command: python proof_read_results.py. 
A window with the recorded signal and the detected events (as orange dots) will appear. A navigation bar on top of the image allows to navigate through the trace, zoom in or zoom out. To add a missed event (false negative) press the keyboard button ‘a’, to delete a wrongly detected event (false positive) press the keyboard button ‘d’. Once done, press ‘u’ to update the results.xlsx file. Go to the results.xlsx file and check if now the manually revised column indicates ‘yes’ for the revised recording.
