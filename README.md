# Spontaneous and miniatures postsynaptic currents (PSCs) detection

### This repository provides an algorithm, primarily based on deep learning, to automate the detection of miniatures and spontaneous inward postsynaptic currents (PSCs) in neurophysiological patch-clamp recordings.

Here below an example of the performance of the algorithm on a voltage clamp recording. The blue dots represents the spontaneous PSCs detected by the algorithm. 

![Alt text](/example.jpg?raw=true)

## Prerequisites
Anaconda or miniconda and python 3. We recommend to use python 3.10 and to create a virtual environment as follows: *conda create --name my_env python=3.10*, where, my_env is a name of your choice. Activate the created environment for the installation of the required packages (see installation section) and for the usage of the algorithm with: *conda activate my_env*.

## Installation
1) install tensorflow; you can try one of the following commands (depending on your system): 
  * *pip install tensorflow*;
  * *conda install tensorflow*;
  * *conda install -c conda-forge tensorflow*. 
  ** Alternatively, go to: https://www.tensorflow.org/install/pip and follow the instructions.
 (Optional) As suggested in the TensorFlow documentation, you can verify the CPU setup by typing: *python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"*. If TensorFlow was installed successfully, this command will return a tensor (tf.Tensor). You can also test the GPU setup (if you have a GPU on your machine) by typing: *python3 -c "import tensorflow as tf; tf.config.list_physical_devices('GPU'))"*. If TensorFlow was installed successfully, this command will return a list of GPU devices.

2) Clone the repository on your local machine with the following command: *git clone https://github.com/Imbrosci/spontaneous-postsynaptic-currents-detection.git*. 

3) make sure that you have pip upgraded with: *pip install --upgrade pip*.

4) Navigate into the cloned repository, spontaneous-postsynaptic-currents-detection and:
 4.1) install the additionally required packages with: *pip install -r requirements.txt*;
 4.2) create two directories, named recordings and results, with: *mkdir recordings results*.

## Preliminary steps before starting the analysis 
Before starting the analysis, there are two simple, but important, steps to do.
1) move the recordings to be analysed in the directory recordings. The recordings should either be abf or txt files.
2) fill the metadata.xlsx file. This file can be found in the directory metadata of the repository.
  Each column of the metadata.xlsx file should be filled as explained below:
  * **Name of recording:** name of the file with the recording including the extension (.abf or .txt). The files listed should be found in the recording directory. 
  * **Channels to use:** if your recording file contains only one channel, enter 1; in case your recording file contains more channels, you should indicate which channels should be analysed. If you want to analyse more channels of a file you can enter the name of the channels separated by a comma or you could keep one row per channel. The algorithm will number the channels found in the recording file in ascending order starting from 1.
  * **Sampling rate (Hz):** sampling rate in Herz.
* **Analysis start at sweep (number):** sweep from which the analysis should start. If your recordings do not have multiple sweeps, select 1. If the recordings have multiple sweeps, the sweeps should have equal length.
* **Cut sweeps first part (ms):** this option offers the possibility to cut out, from the mPSCs / sPSCs analysis, a portion of trace at the beginning of each sweep. This may be relevant if a test pulse (to check, for instance, the stability of a patch clamp recording), or another kind of stimulation, was delivered at the beginning of each sweep. For example, to remove the first half a second of trace from the beginning of each sweep, enter 500.
* **Cut sweeps last part (ms):** similarly, to the option above, this option offers the possibility to cut out a portion of trace from the end of each sweep backward. For example, to remove the last half a second of trace at the end of each sweep, enter 500.
* **Analysis length (min):** enter how many minutes of recording should be analysed. Be aware that analysing many minutes at once (especially if a recording was acquired with a high sampling rate) may cause difficulties in displaying or proof-reading the results. If the recording is shorter than the time indicated in this option, the algorithm will just analyse the recordings until the end. 

## Starting the analysis
To start the recording, navigate into the repository directory and type: *python running_analysis.py*. If everything run correctly some details about the analysis will be printed shortly after.
If there is already a results.xlsx file produced by a former analysis the program will ask you the permission to overwrite the results. To do not overwrite the previous results, just change the name of the results.xlsx file or move it into another directory. 

## Excel file containing the results
Once the results are finished, a results.xlsx file will be generated and can be found under the directory results. 
In the results.xlsx file, there will be a sheet for each analysed recording_channel and summary results sheet.
The sheets for single recording_channel report **the x and y position** (in ms and pA, respectively), **the interevent interval** (between event + 1 and event) (in ms) and **the amplitude** (in pA) for each detected event.
The summary results sheet contain the following information for each recording:
* **Recording filename**;
* **Channel**;
* **Sampling rate (Hz)**;
* **Analysis start at sweep (number)**;
* **Cut sweeps first part (ms)**;
* **Cut sweeps last part (ms)**;
* **Analysis length (sec)**, this may be shorter than specified in the metadata if the recording was shorter;
* **Average interevent interval (ms)**;
* **Average amplitude (pA)**;
* **Average 10-90% rise time (ms)**, calculated from the mean signal of all detected event;
* **Average 90-10% decay time (ms)**, calculated from the mean signal of all detected event;
* **Stdev of the baseline signal (pA)**, Stdev stays for standard deviation and it is calculated from the trace after excluding the detected events, it may provide an information about the quality of the recording and it can be used to exclude certain recordings from the analysis, if, for example, the average amplitude is not larger than 3 times the Stdev);
* **Manually revised**, specified if the results were manually corrected or not, (see proof-read results section for details).

## Check results
To check the quality of the results use the command: *python display_results.py*. The program will ask to enter the file name (including extension) and the channel to display.
Two windows will appear. The first window displays the recorded signal and the detected events (as blue dots), the second window represents the average signal from all detected events. The rise (10-90%) and the decay (90-10%) time will be also displayed as cyan and pink dots, respectively. 

## Proof-read results
If the results are not satisfactory, it is also possible to revise them. 
To correct the results (add false negatives or delete false positives), use the command: *python proof_read_results.py*. As to check the results, the program will ask to enter the file name and the channel to revise. 
A window with the recorded signal and the detected events (as orange dots) will appear. A navigation toolbar on top or on the bottom of the image allow you to navigate through the trace. To add a missed event (false negative) press the keyboard button ‘a’, to delete a wrongly detected event (false positive) press the keyboard button ‘d’. If the revision works, an added datapoint should appear in blue, while a deleted datapoint should turn into red. On Mac, it may be necessary to click the left button of the mouse just after pressing 'a' or 'd' for the revision to take place. Once the proof-reading is done, press ‘u’ to update the results.xlsx file. Wait a few moments before closing the window. Once the update is finished, the statement "The results.xlsx file has been updated" should be printed. Now you can close the window and check if, in the summary results sheet of the results.xlsx file, the "Manually revised" column indicates ‘yes’ for row corresponding to the revised recording.
