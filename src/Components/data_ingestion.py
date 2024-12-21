import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pymongo import MongoClient
from zipfile import Path
from src.Exceptions import CustomException
from src.Logger import logging
from src.Constant import *
from src.Utils.main_utils import MainUtils


@dataclass
class DataIngestionConfig:
    artifact_folder: str = os.path.join(artifact_folder)


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.Utils = MainUtils()


    def export_collection_as_dataframe(self,collection_name,db_name):
        try:
            logging.info('Fetching the dataframe from mongodb data base')
            mongo_client=MongoClient(MONGO_DB_URL)
            collection=mongo_client[db_name][collection_name]

            df=pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df=df.drop(columns=["_id"],axis=1)
            df.replace({"na":np.nan},inplace=True)
            return df
        
        except Exception as e:
            logging.info('Exception occured in  fn:export_collection_as_dataframe')
            raise CustomException(e,sys)
        

    def export_data_into_feature_store_file_path(self)->pd.DataFrame:
        try:
            logging.info(f"Exporting data from mongodb")
            raw_file_path=self.data_ingestion_config.artifact_folder
            os.makedirs(raw_file_path,exist_ok=True)

            logging.info("creating a variable and calling the fn:export_collection_as_dataframe")
            sensor_data=self.export_collection_as_dataframe(collection_name=MONGO_COLLECTION_NAME,
                                                            db_name=MONGO_DATABASE_NAME)
            
            logging.info(f"saving exported data into feature store file path:{raw_file_path}")
            feature_store_file_path=os.path.join(raw_file_path,'wafer_fault.cvs')
            sensor_data.to_csv(feature_store_file_path,index=False)

            return feature_store_file_path
        
        except CustomException as e:
            logging.info('Exception occured in fn:export_data_into_feature_store_file_path')
            raise CustomException(e,sys)
        

    def initiate_data_ingestion(self)->Path:
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")
        try:
            feature_store_file_path=self.export_data_into_feature_store_file_path()
            logging.info('got the data from mongodb')
            logging.info('Exited initiate_data_ingestion method of DataIngestion class')

            return feature_store_file_path
        
        except Exception as e:
            logging.info('Exception occured in fn:initiate_data_ingestion')
            raise CustomException(e,sys) from e