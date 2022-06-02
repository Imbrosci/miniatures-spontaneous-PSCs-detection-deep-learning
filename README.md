# Spontaneous and miniatures postsynaptic currents (PSCs) detection

### This repository provides a deep learning model to automate the detection of miniatures and spontaneous inward postsynaptic currents (PSCs) in neurophysiological patch-clamp recordings.

Here below an example of the performance of the algorithm on a voltage clamp recording. The blue dots represents the spontaneous PSCs detected by the algorithm. 

![Alt text](/example.jpg?raw=true)

The model is currently under revision. Once the revision will be finished, a detailed explanation on how to use it will follow. 
Last update: 02.06.22

## Prerequisites
Anaconda or miniconda and python 3. We recommend to use python 3.9 and to create a virtual environment as follows: conda create --name my_env anaconda python=3.9.

## Installation
Clone the repository with the following command: git clone https://github.com/Imbrosci/miniatures-spontaneous-PSCs-detection-deep-learning.git.
Navigate to the cloned repository miniatures-spontaneous-PSCs-detection-deep-learning and install the required libraries with the following command: pip install -r requirements.txt.
Create the folders recordings and results with the following commands: mkdir recordings and mkdir results.


