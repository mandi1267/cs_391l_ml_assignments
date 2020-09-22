# Assignment Description
This folder contains work for CS 391L assignment 2 (see https://www.cs.utexas.edu/~dana/MLClass/ica-hw.html) for Amanda Adkins.

The objective of this assignment is to use ICA to recover linearly combined signals. 

# Execution Instructions

python3 assign2.py --sound_file_name <location of the larger data set file> --small_dataset_file_name <location of the file containing the small data set and the mixing matrix for the small data set>

Files to use are located here: https://www.cs.utexas.edu/~dana/MLClass/ica-hw.html

This runs an abbreviated version of the experiments run for the report. The full evaluation can be run by removing the 
configuration overrides under NOTE in the 3 experiment execution functions (search NOTE in assign2.py). 

# Required Libraries
You will need to have python3 to run this. In addition, you'll need to have numpy, scipy, and matplotlib which can be installed using:
"python3 -m pip install --user numpy scipy matplotlib"
