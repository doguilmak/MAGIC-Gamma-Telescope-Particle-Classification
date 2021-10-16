
# Predict Gamma (signal) and Hadron (background) from MAGIC Gamma Telescope with XGBoost and ANN

## Problem Statement

The purpose of this study is to predict the result of the telescope observation. Algorithm is trying to predict the class (11th row) which is gamma (signal) or hadron (background). 

## Dataset

Dataset is downloaded from [archive.ics.uci.edu](https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope) website. You can find the details of the dataset in that website. Dataset has **11 columns** and **19020 rows without the header**.

## Methodology

In this project, as stated in the title, results were obtained through **XGBoost** and **artificial neural networks**.  You are free to visit [XGBoost](https://xgboost.ai/) website for learn the XGBoost method better.

## Analysis

 | # | Column | Non-Null Count | Dtype |
|--|--|--|--|
| 0 | 0 | 10000 non-null | float64
| 1 | 1 | 10000 non-null | float64
| 2 | 2 | 10000 non-null | float64
| 3 | 3 | 10000 non-null | float64
| 4 | 4 | 10000 non-null | float64
| 5 | 5 | 10000 non-null | float64
| 6 | 6 | 10000 non-null | float64
| 7 | 7 | 10000 non-null | float64
| 8 | 8 | 10000 non-null | float64
| 9 | 9 | 10000 non-null | float64
| 10 | 10 | 10000 non-null | object

**dtypes: float64(10), object(1)**

### XGBoost
***Confusion Matrix(XGBoost):***

| 3807 | 481 |
|--|--|
| **251** | **1700** |


> **Accuracy score(XGBoost): 0.8826735053694502**
> 
> **Process took 0.8865966796875 seconds.**

### Artificial Neural Network

| Layer (type) | Output Shape | Param # |
|--|--|--|
| dense_14 (Dense) | (None, 16) | 176 |
| dense_15 (Dense) | (None, 32) | 544 |
| dense_16 (Dense) | (None, 16) | 528 |
| dense_17 (Dense) | (None, 1) | 17 |

Total params: 1,265
Trainable params: 1,265
Non-trainable params: 0

In ***Plot*** folder, you can find ***model_val_accuracy.png***  which is showing plot of test and train accuracy with val_accuracy. Accuracy values and also plot can change a bit after you run the algorithm.

Model predicted class as [[0]].

> **Process took 1.8856332302093506 seconds.**

## How to Run Code

Before running the code make sure that you have these libraries:

 - pandas 
 - time
 - sklearn
 - numpy
 - warnings
 - xgboost
 - keras
    
## Contact Me

If you have something to say to me please contact me: 

 - Twitter: [Doguilmak](https://twitter.com/Doguilmak).  
 - Mail address: doguilmak@gmail.com
 
