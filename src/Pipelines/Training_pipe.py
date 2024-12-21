import os
import sys

from src.Exceptions import CustomException
from src.Logger import logging
from src.Components.data_ingestion import DataIngestion
from src.Components.data_transformation import DataTransformation
from src.Components.model_trainer import ModelTrainer


class TrainingPipeline:

    def start_data_ingestion(self):
        try:
            logging.info('Data Ingestion pipeline started')
            data_ingestion=DataIngestion()
            feature_store_file_path=data_ingestion.initiate_data_ingestion()
            return feature_store_file_path
        
        except Exception as e:
            logging.info('Exception occured in fn:start_data_ingestion')
            raise CustomException(e,sys)
        

    def start_data_transformation(self,feature_store_file_path):
        try:
            logging.info('Data transformation pipeline started')
            data_transformation=DataTransformation(feature_store_file_path=feature_store_file_path)
            train_arr, test_arr, preprocessor_path=data_transformation.initiate_data_transformation()
            return train_arr,test_arr,preprocessor_path
        
        except CustomException as e:
            logging.info('Exception occured in fn: start_data_transformation')
            raise CustomException(e,sys)
        

    def start_model_training(self, train_arr, test_arr):
        try:
            logging.info('Model training pipeline started')
            model_trainer=ModelTrainer()
            model_score=model_trainer.initiate_model_trainer(train_arr,test_arr)
            return model_score
        
        except CustomException as e:
            logging.info("exception occured in fn:start_model_trainer")
            raise CustomException(e,sys)
        
    def run_TrainingPipeline(self):
        try:
            logging.info('Training Pipeline has been initiated')
            feature_store_file_path=self.start_data_ingestion()
            train_arr,test_arr,preprocessor_path=self.start_data_transformation(feature_store_file_path)
            r2_square=self.start_model_training(train_arr,test_arr)
            logging.info(f'Training Pipeline completed. Trained model score is {r2_square}')

        except CustomException as e:
            logging.info("exception occured in  fn:run_TrainingPipeline")
            raise CustomException(e,sys)