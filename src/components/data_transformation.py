import sys
import os
import dataclasses from dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        '''
            This function is responsible for data transformation
        '''
        try:
            numerical_column = ['writing_score', 'reading_score']
            categorical_column = [
                'gender', 
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
                ]
            ''' 
                create a pipeline for the data transformation. Create 2 pipeline (numerical and categorical)
                1). Handle the missing value - imputer: statergy median
                2). Standard Scaler - 
                3). One HotEncoding - this is for the categorical value
            '''
            # numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    {'imputer', SimpleImputer(strategy="median")},
                    {'scaler', StandardScaler()}
                ]
            )
            # categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    {'imputer', SimpleImputer(strategy='most_frequent')},
                    {'one_hot_encoding', OneHotEncoder()},
                    {'scaler',StandardScaler()}
                ]
            )
            # logging
            logging.info("Numerical columns encoding is completed")
            logging.info("Categorical columns encoding is completed")
            # to combine this two pipeline we use columnTransformer
            preprocessor = ColumnTransformer(
                [
                    {"num_pipeline":num_pipeline, numerical_column},
                    {"cat_pipeline":cat_pipeline, categorical_column}
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(sys, e)