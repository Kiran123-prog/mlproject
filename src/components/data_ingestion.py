# import the libraries
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

# class of data ingestion
class DataIngestion:
    # 2. initialize the data ingestion - setup the artifacts of data set train, test, raw. 
    def __init__(self):
        # to initialize the artifacts - create an object
        self.ingestion_config = DataIngestionConfig()
    # 3. initiate data ingestion - read the data set    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or components")
        try:
            # read the csv data as dataframe
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the dataset as dataframe")

            # make a directory from the artifacts 
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok= True)
            # save the dataframe as csv into the raw data path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set =  train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)


            logging.info("Ingestion the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(sys, e)
            

if __name__ == "__main__":
    # 1. start Data Ingestion
    Obj = DataIngestion()
    # return the train and test data
    train_data, test_data = Obj.initiate_data_ingestion()

    # 2. initiate data transformation  
    dataTransformation_obj = DataTransformation()
    # return the transformed train and test array. And the last one is pkl file we don't want this
    train_arr, test_arr, _ =  dataTransformation_obj.initiate_data_transformation(train_data, test_data)

    # 3. start the model training
    modelTrainer_obj = ModelTrainer()
    r2_score = modelTrainer_obj.initiate_model_trainer(train_arr, test_arr)
    print(r2_score)
