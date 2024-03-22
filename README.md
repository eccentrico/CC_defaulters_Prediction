# CC_defaulters_Prediction

# Data Science Project: Preliminary Data Exploration

This Jupyter notebook contains a comprehensive preliminary data exploration for a data science project. The notebook is structured to facilitate understanding of the data, enable initial analysis, and prepare for further in-depth study and model building.

## Notebook Overview

- **Total Cells**: 55
  - **Code Cells**: 34
  - **Markdown Cells**: 21

The notebook is divided into sections, starting with markdown explanations followed by code implementations. It begins with an introduction to the preliminary data exploration process.

## Features

1. **Data Import and Cleaning**: Initial steps involve importing necessary libraries and cleaning the dataset for analysis.
2. **Exploratory Data Analysis (EDA)**: This section includes visualization and statistical analysis to understand the data better.
3. **Model Preparation**: Data is prepared for modeling, including train-test split and preprocessing steps.
4. **Model Building**: Several models are built and evaluated, including Random Forest Classifier, Linear Regression, and Gradient Boosting Classifier, among others.
5. **Evaluation**: The models are evaluated using various metrics such as ROC-AUC score, accuracy score, R-squared, and confusion matrix.

## First Code Snippet

The notebook starts with the import of essential Python libraries for data analysis and machine learning:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
