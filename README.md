# ALMS-CW
The repository of Applied machine learning system assignment. repository link: https://github.com/Xingminliiii/ALMS-CW.git 

# A brief description of the organization of project
- Task A in this study uses a 'Simple MNIST convnet,' a binary classification model consisting of three convolutional layers with a (2 × 2) filter size and two fully connected layers, to diagnose pneu- monia using the 'PneumoniaMNIST' dataset.
- Task B in this study engaged 'Architecture 3,' a multi- class classification model originally designed for the MNIST and Fashion MNIST datasets, to categorize the 'PathMNIST' dataset into nine distinct classes. Differing from the 'Simple MNIST convnet', 'Architecture 3' inte- grates an additional dropout layer following each convolu- tional layer and uses a softmax output for class categoriza- tion. 
- Learning curve for each Task is plotted. 
- Statistics for Task A is calculated, including: test accuracy, precision, recall, F1-score  and a kappa statistic, to evaluate Task A. 
- Extra confusion matrix and ROC curve is generated to evaluate the performance of Task B. 

# the role of each file
## A
- '__init__.py': mark a directory as a Python package directory.
- taskA.py: A script inclues all of the modules should be imported when perform and evaluate CNN model in Task A 
## B
- '__init__.py': mark a directory as a Python package directory.
- taskA.py: A script inclues all of the modules should be imported when perform CNN model in Task B
- taskA.py: A script inclues all of the modules should be imported when evaluate CNN model in Task B
## main.py
file to excecute the code
## requirements.txt
All of the libraries required to implement the code

# Machine Learning Project Setup

## Installation

To ensure that you have the correct versions of libraries needed for this project, a `requirements.txt` file has been created. Please install the dependencies by executing the following command in your terminal:

`pip install -r requirements.txt`

## Running the Code

After installing the required libraries, you can run the code by typing the following command in your terminal:

`python main.py` 


## Requirements

The `requirements.txt` file includes the following libraries:

- numpy
- matplotlib
- keras
- tensorflow
- scikit-learn

Please verify that these libraries are installed by checking the `requirements.txt` file.

## Execution Time Note

Running `main.py` may take around 40 minutes due to the time required for model training.

### For Quick Testing

If you want to check if the code executes without performing full training, consider reducing the number of epochs for Task B to 2 or 3.

## Dataset Information

The `PneumoniaMNIST` and `PathMNIST` datasets have been pre-split into training, validation, and testing sets, so there is no need for additional dataset splitting.


