# How to Run the Project
The project entry point is `main_cer.py`. If you run this file using `python main_cer.py` it should train the model and and run the validation and test sets on the trained model. Right now it only caculates accuracy. 

To successfully run the code with [neptune](https://neptune.ai/), you need to set the `NEPTUNE_API_TOKEN` as an environment variable. [Here](https://docs.neptune.ai/api/environment_variables/#) you can see how to do that. 

If you have access to the [unity](https://unity.rc.umass.edu/) servers and want to run this project as a batch there, the script is given in [sbatch.run.sh](https://github.com/hheickal/code-edit-representation/blob/master/sbatch.run.sh). You have to create your own conda environment though using the [requirements.txt](https://github.com/hheickal/code-edit-representation/blob/master/requirements.txt) to `source` the conda environment inside the `sbatch.run.sh`.

# Dataset
The dataset used in the project comes from [CSEDM](https://sites.google.com/ncsu.edu/csedm-dc-2021/dataset). The dataset is preprocessed and put in `data/dataset.pkl`. The columns in the dataset are:
- `problemID`: The id of the problem in CSEDM dataset.
- `problemDescription`: The textual descrtiption of the problem.
- `studentID_1`: The student ID in CSEDM dataset of the first student.
- `test_case_verdict_i_1` The test case verdicts of the code submission indexed using `i_1`. This is a string containing multiple 0-3. 0 means `correct`, 1 means `wrong answer`, 2 means `run-time error`, 3 means `time-limit exceeded` in test_case_verdict_x_y where x can be ['i','j'] and y can be [1,2].

- `codeID_i_1`: codeID of `i_1` in CSEDM dataset. This is the chronologically earlier submission made by `studentID_1` in `problemID`.
- `code_i_1`: The actual code in texts of  `codeID_i_1`.
- `score_i_1`: The assigned score in CSEDM dataset for `codeID_i_1`.
- `score_calc_i_1`: Calculated score by our score against the test cases for `codeID_i_1`.
- `test_case_verdict_j_1`: The test case verdicts of the later code submission indexed using `j_1` by the student `studentID_1`.
- `codeID_j_1`: Same as earlier.
- `code_j_1`: Same as earlier.
- `score_j_1`: Same as earlier.
- `score_calc_j_1`: Same as earlier.
- `studentID_2`: Same as student 1.
- `test_case_verdict_i_2`: Same as student 1.
- `codeID_i_2`: Same as student 1. 
- `code_i_2`: Same as student 1. 
- `score_i_2`: Same as student 1. 
- `score_calc_i_2`: Same as student 1.
- `test_case_verdict_j_2`: Same as student 1.
- `codeID_j_2`: Same as student 1.
- `code_j_2`: Same as student 1. 
- `score_j_2`: Same as student 1.
- `score_calc_j_2`: Same as student 1.
- `is_similar`: Binary label (True/False) denoting whether the change between (`code_i_1`, `code_j_1`) and (`code_i_2`, `code_j_2`) is similar or not. 

# High-level TO DO
* Prepare the data to include boundary cases for negative examples 
* Prepare the model to handle batched data so that we can share weights among the encoders instead of copying them to for 4 inputs. 
* Do the gradient accumulation.
* Explore diffrent loss functions. 

# Code Structure
## `preprocess_data.ipynb`
The puropose of this notebook is to preprocess the CSEDM dataset and prepare it for the CER project. We have the test cases for 17 of the problems. We also have the execution verdict of the runs for this test cases in a separate file. We put these two together and created `data/dataset.pkl`.
### Requirements
- Each dataset row contains a two pair of codes.
- The student id for both the codes should be the same. 
- The code id must be different. 
- We also need the scores for both codes
- There should also be the test case masks for both codes these are
- The testCaseMask of the two codes must be different
- The dataset should contain only the codes with test cases (17 problems so far)
- The pair of codes in each row can be either consecutive chronologically or not. Based on this we can create different datasets. 

## `main_cer.py`
Main entry point for the project. This file prepares the training and testing data, and runs the training and testing loop.

## `configs_cer.yaml`
Holds all the config values for various model, data, training and testing options. 

## `data_loader.py`
Data processing functions. 

## `model.py`
Defines all the models. 

## `trainer.py`
Defines all the training related helper functions

## `eval.py`
Defines all the functions do the evaluation of the test outputs.

## `utils.py`
Defines all the helper utility functions.
