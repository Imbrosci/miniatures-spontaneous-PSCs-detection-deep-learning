# Spontaneous and miniatures postsynaptic currents (PSCs) detection

### This repository provides a deep learning model to automate the detection of miniatures and spontaneous inward postsynaptic currents (PSCs) in neurophysiological patch-clamp recordings.

Here below an example of the performance of the algorithm on a voltage clamp recording. The blue dots represents the spontaneous PSCs detected by the algorithm. 

![Alt text](/example.jpg?raw=true)

The model is currently under revision. Once the revision will be finished, a detailed explanation on how to use it will follow. 
Last update: 05.06.22

## Prerequisites
Anaconda or miniconda and python 3. We recommend to use python 3.10 and to create a virtual environment as follows: conda create --name my_env anaconda python=3.10.

## Installation
Clone the repository on your local machine with the following command: git clone https://github.com/Imbrosci/spontaneous-postsynaptic-currents-detection.git. 
Navigate to the cloned repository, spontaneous-postsynaptic-currents-detection and:
1) install the required libraries with: pip install -r requirements.txt;
2) create two folders, named recordings and results, with: mkdir recordings and mkdir results.


