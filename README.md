# High-level TO DO
* Prepare the data to include boundary cases for negative examples 
* Prepare the model (first version ready)
* Do the gradient accumulation 
* Train
* Test
* What is omegaConf?

# Code Structure
## `preprocess_data.ipynb`
The puropose of this notebook is to preprocess the CSEDM dataset and prepare it for the CER project. We have the test cases for 17 of the problems. We also have the execution verdict of the runs for this test cases in a separate file. We need to put these two together to generate a final dataset for the project. We will name it `dataset_cer.pkl`.
### Requirements
- Each dataset row should contain a pair of codes. 
-- The student id for both the codes should be the same. 
-- The code id must be different. 
-- We also need the scores for both codes
-- There should also be the test case masks for both codes
-- The testCaseMask of the two codes must be different
- The dataset should contain only the codes with test cases (17 problems so far)
- The pair of codes in each row can be either consecutive chronologically or not. Based on this we can create different datasets. 

## `main_cer.py`
Main entry point for the project. This file prepares the training and testing data, and runs the training and testing loop.
### To Do
- Update the file to match the current project.
### Works Done
- Nothing so far
## `configs_cer.yaml`
Holds all the config values for various model, data, training and testing options. 
### To Do
- Update the file to match the current project.
### Works Done
- Nothing so far
## `data_loader.py`
Data processing functions. 
### To Do
- Update the file to match the current project.
### Works Done
- Nothing so far
## `model.py`
Defines all the models. 
### To Do
- Update the file to match the current project.
### Works Done
- Nothing so far
## `trainer.py`
Defines all the training related helper functions
### To Do
- Update the file to match the current project.
### Works Done
- Nothing so far
## `eval.py`
Defines all the functions do the evaluation of the test outputs.
### To Do
- Update the file to match the current project.
### Works Done
- Nothing so far
## `utils.py`
Defines all the helper utility functions.
### To Do
- Update the file to match the current project.
### Works Done
- Nothing so far
## `run.py`
This file is used to run different experiment, I guess. Not sure if this will be required for our project.
## `interpret.py`
This file is used to interpret the generated codes in info-okt. Not sure if we need this.