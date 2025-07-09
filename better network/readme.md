# File Structure

root
- **backend**
  - config.py: This file contains the main parameters of both the raw data and the reconstruction.
  - data_initializer.py: This file loads the data from the path specified in config.py and converts it to a torch
    tensor. Eventually, this will also have the code for processing hardware ODMR data. 
  - forward_transformation.py: This file contains all the code for converting between stray fields and magnetizations
  - model.py: This is the CNN that tries to identify the magnetization
  - utils.py: This contains some code for plotting and switching between numpy arrays and torch tensors
- **data** just contains a bunch of test data
- debugging.py: this is for my own testing
- train.py: this is where I run the model. 

# Notes
The issue I'm running into right now is that my training loss gets stuck at the initial value whenever I try using the 
forward transformed stray field. When I tell the model to directly reproduce a given dataset, it does a good job which 
indicates that its at least somewhat functional. Also, I've noticed that scaling the input stray field by a large 
constant (like 1000) tends to produce much better reproductions. 