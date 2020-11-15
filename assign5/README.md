# Assignment Description
This folder contains work for CS 391L assignment 5.

The objective of this assignment is to use modular RL to learn policies for having an agent navigate a grid world, while moving forward in the world, avoiding obstacles, trying to stay on the sidewalk, and picking up litter.  

# Execution Instructions

Run the assignment code using:
python3 assign5.py 

# Code Organization
The purposes of each of the files are as follows:
- assign5.py - orchestrates program execution and creates custom and random grids.
- grid_world.py - Contains representation of world including state of the agent.
- grid_rep.py - Contains representation of world excluding state of the agent.
- rewarder.y - Contains reward calculation objects for each module
- states.py - Contains state representation for each module.
- state_retrievers.py - Contains objects for retrieving module states from the grid world
- plotting.py - Contains function for plotting the grid world with different options
- q_learning.py - Contains functions and classes related to Q learning and executing a test run.  
   

# Required Libraries
You will need to have python3 to run this. In addition, you'll need to have numpy, scipy, joblib, and matplotlib, which can be installed using:
"python3 -m pip install --user numpy scipy matplotlib joblib"
