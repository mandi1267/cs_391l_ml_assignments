# Assignment Description
This folder contains work for CS 391L assignment 4.

The objective of this assignment is to use Gaussian processes to predict the positions of markers on a subject tracing 
a curve in VR. We also investigated how differences in hyperparameters over time may indicate different muscle states. 

# Execution Instructions

To run, ensure that the data files are in the same directory as assign4.py and then run:
python3 assign4.py

The code currently runs on marker 37 and coordinate z of the CJ data set and only uses one value for each sliding
 window parameters. It can be modified to do a more exhaustive search or use different data. 

# Code Organization
The purposes of each of the files are as follows:
- assign4.py - orchestrates program execution by reading data, creating model, iterating over sliding windows, and 
evaluating model performance.
- gaussian_process_predictor.py - Provides interface for training Gaussian process models and predicting values.  
   

# Required Libraries
You will need to have python3 to run this. In addition, you'll need to have numpy, scipy, matplotlib, and scikit-learn, which can be installed using:
"python3 -m pip install --user numpy scipy matplotlib joblib scikit-learn"
