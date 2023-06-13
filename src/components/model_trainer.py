import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config =  ModelTrainerConfig()

    # Initiate the model training with preprocessed train and test data
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split the training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:,-1],
                test_arr[:, :-1],
                test_arr[:,-1]
            )

            # create the dictionaries of models that we are going to train
            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regressor": LinearRegression(),
                "KNeighbors Classifier": KNeighborsRegressor(),
                "XGBoostClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(),
                "AdaBoost Classifier": AdaBoostRegressor()
            }

            # make parameter for hyper-parameter tuning
            params = {
                "Decision Tree" : {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],                
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    # 'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },               
                "Linear Regressor":{},
                "KNeighbors Classifier":{
                    'n_neighbors':[5,7,9,11],
                    # 'weights':['uniform','distance'],
                    # 'algorithm':['ball_tree','kd_tree','brute']
                },
                "XGBoostClassifier":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Classifier":{
                    'depth': [6,8,10],
                    # 'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },                
                "AdaBoost Classifier":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }

            }

            # Evaluate our models - without parameter tuning - this method present inside the utils file
            '''
            model_report:dict=evaluate_model(X_train = X_train, y_train = y_train, X_test =X_test, 
                                                y_test = y_test, models = models)
            '''

            # Evaluate our models - with parameter tuning - this method present inside the utils file
            model_report:dict=evaluate_model(X_train = X_train, y_train = y_train, X_test =X_test, 
                                                y_test = y_test, models = models, params = params)

            # To get best model score from the dict : models
            best_model_score = max(sorted(model_report.values()))

            # to get best model name form the dict : models
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # to get the best model
            best_model = models[best_model_name]

            # filter out worst performance models
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found in the train and test dataset")

            '''
            save the model path call save_object/ trained model file path. and give the best model
            as our pickle file
            '''
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, #model.pkl
                obj=best_model
            )

            # find the r2_score of the best_model with the test data(X_test)
            predicted_test = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted_test)

            return r2_square


        except Exception as e:
            raise CustomException(sys, e)
        
        
