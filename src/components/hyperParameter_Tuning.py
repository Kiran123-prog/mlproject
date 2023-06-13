import numpy as np
import pandas as pd

import os
print("hello world")

for dirname, _, filenames in os.walk('notebook\data\TitanicDataset'):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        print(file_path)

from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

print("let's start")

# Load data
train_csv = pd.read_csv("notebook/data/TitanicDataset/train.csv")
test_csv = pd.read_csv("notebook/data/TitanicDataset/test.csv")



# Train, test split
y_train = train_csv['Survived'].copy()
x_train = train_csv.drop(['Survived', 'PassengerId', 'Name', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)

print(y_train)