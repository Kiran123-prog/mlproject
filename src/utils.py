import numpy as np
import pandas as pd
import os
import sys
import dill
import pickle

from sklearn.metrics import r2_score

from src.exception import CustomException

# to save the object file
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_object:
            pickle.dump(obj, file_object)

    except Exception as e:
        raise CustomException(e, sys)


# model evaluation
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            # fit it/train and test data to our model
            model.fit(X_train,y_train)
            # do the prediction on X_train
            X_train_pred =  model.predict(X_train)
            # do the prediction on X_test
            X_test_pred = model.predict(X_test)
            # find out the r2 score on each predition - for train data score prediction
            train_model_score = r2_score(y_train, X_train_pred)
            # for test data score prediction 
            test_model_score = r2_score(y_test, X_test_pred)
            # store the values as key:value pair
            report[list(models.keys())[i]] = test_model_score

        return report
     
    except Exception as e:
        raise CustomException(e, sys)


