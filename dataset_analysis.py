# AI & ML Internship - Task 1
# Understanding Dataset & Data Types
# Dataset Used: Titanic Dataset

import pandas as pd
import numpy as np

# Load Titanic Dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Display first few rows
print("First 5 records:")
print(df.head())

# Display last few rows
print("\nLast 5 records:")
print(df.tail())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Unique values in categorical columns
print("\nUnique values in Sex column:")
print(df['Sex'].unique())

print("\nUnique values in Embarked column:")
print(df['Embarked'].unique())

# Target variable
print("\nTarget Variable: Survived")

"""
OBSERVATIONS:

1. The Titanic dataset contains passenger details such as age, gender, class, and survival status.
2. Numerical features include Age, Fare, SibSp, and Parch.
3. Categorical features include Sex, Embarked, Ticket, and Cabin.
4. The target variable is Survived, which is binary (0 or 1).
5. Missing values are present in Age, Cabin, and Embarked columns.
6. The dataset is suitable for classification problems.
"""
