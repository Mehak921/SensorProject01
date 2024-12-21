import os
import sys
import shutil
import pickle

import pandas as pd
from flask import request
from dataclasses import dataclass

from src.Exceptions import CustomException
from src.Logger import logging
from src.Constant import *
from src.Utils.main_utils import MainUtils


@dataclass
class PredictionPipelineConfig:
    prediction_output_dirname:str="prediction_output"
    prediction_file_name:str="predicted_file.csv"
    model_file_path:str=os.path.join(artifact_folder,"model.pkl")
    preprocessor_path:str=os.path.join(artifact_folder,"preprocessor.pkl")
    prediction_file_path:str=os.path.join(prediction_output_dirname,prediction_file_name)


class PredictionPipeline:
    def __init__(self, request: request): # type: ignore
        self.request = request
        self.Utils = MainUtils()
        self.prediction_pipeline_config = PredictionPipelineConfig()


    def save_input_file(self)->str:
        try:
            logging.info('Creating the file path')
            pred_file_input_dir="prediction_uploads"
            os.makedirs(pred_file_input_dir,exist_ok=True)

            input_csv_file=self.request.files['file']
            pred_file_path=os.path.join(pred_file_input_dir,input_csv_file.filename)
            input_csv_file.save(pred_file_path)

            return pred_file_path
        except CustomException as e:
            logging.info('Exception occured in fn:save_input_file')
            raise CustomException(e,sys)
        

    def predict(self,features):
        try:
            logging.info('Predicting the output')
            model=self.Utils.load_object(self.prediction_pipeline_config.model_file_path)
            preprocessor=self.Utils.load_object(self.prediction_pipeline_config.preprocessor_path)
            transformed_x=preprocessor.transform(features)
            pred=model.predict(transformed_x)
            logging.info("Prediction was done sucessfully")
            return pred
        except CustomException as e:
            logging.info('Exception occured in fn:predict')
            raise CustomException(e,sys)
        

    def get_predicted_dataframe(self,input_dataframe_path:pd.DataFrame):
        try:
            logging.info('Dataframe has been received from the user')
            prediction_column_name:str=TARGET_COLUMN
            logging.info('Dataframe is being converted to pandas dataframe')
            input_dataframe:pd.DataFrame=pd.read_csv(input_dataframe_path)
            logging.info('Cleaning the data')
            input_dataframe=input_dataframe.drop(columns="Unnamed: 0") if "Unnamed: 0" in input_dataframe.columns else input_dataframe
            prediction=self.predict(input_dataframe)
            logging.info('Creating the prediction column in dataframe and decoding the output')
            input_dataframe[prediction_column_name]=[pred for pred in prediction]
            target_column_mapping={0:'bad',1:'good'}
            input_dataframe[prediction_column_name]=input_dataframe[prediction_column_name].map(target_column_mapping)
            logging.info('Saving the predicted dataframe')
            os.makedirs(self.prediction_pipeline_config.prediction_output_dirname,exist_ok=True)
            input_dataframe.to_csv(self.prediction_pipeline_config.prediction_file_path,index=False)
            logging.info('Prediction completed')

        except CustomException as e:
            logging.info('Exception occured in fn:get_predicted_dataframe')
            raise CustomException(e,sys)
        

    def run_PredictionPipeline(self):
        try:
            logging.info('Prediction pipeline has begun')
            input_csv_path=self.save_input_file()
            self.get_predicted_dataframe(input_csv_path)
            logging.info('Prediction pipeline completed')
            return self.prediction_pipeline_config
        
        except CustomException as e:
            logging.info('Exception occured in fn:run_PredictionPipeline')
            raise CustomException(e,sys)