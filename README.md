# RT-Sort
RT-Sort: an action potential propagation-based algorithm for real time spike detection and sorting with millisecond latencies

## Source code organization
This repository contains all of the code used in the development of RT-Sort and used to create the figures in the RT-Sort publication. The source code is divided into 3 main folders. The folders and some highlighted scripts are described below: 
1. ```spike_detection```: Training and validating the DL models used for spike detection.
    - ```spikesort_matlab4.py```: Spike sort (with Kilosort2) and curate recordings used for training and validating the DL model 
    - ```train/single_mea.ipynb``` and ```train/single_neuropixels.ipynb```: The core training and validating of the MEA and Neuropixels models, respectively.
2. ```spike_sorting```: The sequence detection and spike assignment portion of RT-Sort (and some code for validating the DL models).
    - ```run_alg/si_rec13```: Jupyter Notebooks that start with this run the complete RT-Sort algorithm (from sequence detecting to simulated real time spike sorting) on a specific recording or group of recordings.
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
```si_rec13_patch.ipynb```: Ground truth MEA
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
```si_rec13_ground_truth.ipynb```: Simulated mouse in vivo Neuropixels
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
```si_rec13.ipynb```: Mouse in vivo Neuropixels
1. ```additional_analysis```: Additional analyses not included in ```spike_detection``` or ```spike_sorting```, such as Figure 4B-G.

## Installation
In terminal with Anaconda installed, run ```conda env create -f env.yml``` to create the Python environment with all the libraries with the correct versions installed.<br>

If that does not work, install the following Python libraries (ideally with the specified versions):
- h5py==3.7.0
- numpy==1.21.6
- matplotlib==3.5.3
- scipy==1.7.3
- scikit-learn==1.0.2
- torch==1.12.1+cu113
- torch_tensorrt==1.2.0
- diptest==0.6.0
- tqdm==4.65.0
- spikeinterface==0.94.0
- jupyter==1.0.0
- comet-ml==3.36.0

## Running source code
Many scripts act as modules for other scripts. For the scripts in ```src/spike_detection```, ```src``` used in import statements refers to the folder ```src/spike_detection```. For the scripts in ```src/spike_sorting```, ```src``` used in import statements refers to the folder ```src/spike_sorting```.

## Terminology discrepancies
There are some discrepancies in the terminology used in the source code and the RT-Sort publication. The following are the most notable:
- Variations of propagation signal, such as ```prop_signal```, refer to the RT-Sort algorithm.
- ```latencies``` and ```delays``` in the source code refer to "time intervals" and  "latencies" in the RT-Sort publication, respectively.
- ```clusters``` may refer to "preliminary propagation sequences" depending on the context.
- ```SI``` or ```si``` refer to Neuropixels.

## Correspondence
Tjitse van der Molen: tjitse@ucsb.edu<br>
Max Lim: maxlim@ucsb.edu
