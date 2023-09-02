# Customer Churn Prediction in Mobile Communication Company

This repository contains a Python codebase for predicting customer churn in a mobile communication business. Customer churn is defined as customers not using their mobile SIM card for a duration of 30, 60, 90 days, or any period specified by the business owner.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Feature Engineering](#feature-engineering)
6. [Month-on-Month Changes](#month-on-month-changes)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Machine Learning Model](#machine-learning-model)
9. [Implementation](#implementation)
10. [Conclusion](#conclusion)

## Introduction

In this project, we aim to predict customer churn in a mobile communication company. Customer churn is a crucial metric for telecom companies as it directly impacts their revenue and growth. Identifying potential churners and implementing retention strategies can significantly reduce customer loss.

## Getting Started

### Libraries Used

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scipy
- Scikit-learn

### Data Loading

The dataset is loaded from a CSV file named "Prepaid.csv."


## Data Preprocessing
### Data Overview
We begin by exploring the dataset to understand its structure and contents.

### Handling Missing Values
We check for missing values in the dataset and handle them accordingly.

### Data Cleaning
We perform data cleaning by removing features with a single unique value.

## Exploratory Data Analysis (EDA)
Various data visualizations are created to better understand the data. This includes bar plots, pair plots, and box plots to visualize the distribution of churn status and other relevant features.


## Feature Engineering
 The code performs feature engineering by calculating averages and creating flags for various usage patterns, such as call duration, data usage, and more.

## Month-on-Month Changes
To identify any risk associated with churn, the code calculates month-on-month changes for several key features.


## Model building
We build predictive models to identify potential churners.


## Evaluation Metrics
We evaluate model performance using various metrics such as accuracy, precision, recall, and ROC curves. Also, a correlation analysis is conducted to identify features that are highly correlated with churn. These features can be useful for prediction.

## Machine Learning Model
hile not shown in the provided code, a machine learning model would typically be trained using the engineered features and historical churn data to predict future churn.

## Implementation

[Link to Implementation Details](churn_analysis.md)

## Conclusion
In this project, we developed a predictive model to identify potential customer churn in a mobile communication company. By leveraging data preprocessing, exploratory data analysis, and feature engineering, we aimed to improve model accuracy and provide valuable insights for reducing customer churn.

Feel free to explore the code and adapt it to your specific business needs. For more details, refer to the code files in this repository.


Please replace the placeholders with actual content and modify the sections as needed to provide a comprehensive overview of your project.



