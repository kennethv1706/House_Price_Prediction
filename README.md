# House_Price_Prediction
This project aims to predict house prices based on details and features of the house. Using various regression models, it identifies which model gives the best results for predicting the price of a house.

## Table of Contents
1. [Import Libraries](#import-libraries)
2. [Data Loading](#data-loading)
3. [Data Cleaning](#data-cleaning)
4. [Feature Selection](#feature-selection)
5. [Feature Engineering](#feature-engineering)
6. [Model Definition and Model Training](#model-definition-and-model-training)
7. [Model Evaluation](#model-evaluation)
8. [Model Saving](#model-saving)
9. [Libraries Used](#libraries-used)
10. [Link to Model](#link-to-model)
11. [Conclusion](#conclusion)

## 1. Import Libraries
The necessary libraries for data manipulation, visualization, and model training are imported in this step.

## 2. Data Loading
The dataset is loaded from a CSV file or any other format. We use `pandas` to load and inspect the dataset.

## 3. Data Cleaning
In this step, missing values, duplicates, and irrelevant columns are handled.

## 4. Feature Selection
Selecting relevant features for the model to improve prediction performance. Unnecessary or redundant features are removed.

## 5. Feature Engineering

## 6. Model Definition and Model Training
We define multiple regression models and train them on the dataset:
- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Random Forest Regression**
- **Random Forest Regression (RandomizedSearchCV)**
- **Random Forest Regression (GridSearchCV)**
- **Gradient Boosting**
- **Gradient Boosting (GridSearchCV)**

## 7. Model Evaluation
We evaluate the models based on several metrics such as Mean Squared Error (MSE), R², Mean Absolute Percentage Error (MAPE), Root Mean Squared Error (RMSE), and Mean Absolute Deviation (MAD).

## 8. Model Saving
Once the model with the best performance is identified, it is saved for future use.


## Libraries Used
- **Pandas**: For data manipulation
- **NumPy**: For numerical operations
- **Seaborn**: For data visualization
- **Matplotlib**: For plotting graphs
- **Scikit-learn**: For model building and evaluation

## Link to Model
The trained model can be accessed and used on [Hugging Face](https://huggingface.co/spaces/kennethv1706/Price_Prediction).

## Conclusion
- **Linear Regression**, **Ridge Regression**, and **Lasso Regression** performed best across most evaluation metrics (MSE, R², MAPE, RMSE, and MAD). These models provided reliable and accurate results.
- **Gradient Booster Gridsearch** performed exceptionally well in most metrics, especially MSE and R², but did not perform as well in MAPE and RMSE compared to linear models.
- **Random Forest** had poor performance across all evaluation metrics, with higher errors compared to the other models.
