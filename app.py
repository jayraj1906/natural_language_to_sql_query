from fastapi import FastAPI
from fastapi import HTTPException
from src.my_logging.logger import logging
from src.exception.exception import GraphDBProjectException
from src.schema.sql_and_neodb import SQLConfig,Neo4jConfig
from src.schema.getResponseSchema import UserInput
from src.extractAndStore import ExtractAndStoreMetaDataYAML

import sys
import json

import yaml

app=FastAPI()


@app.post("/rag/extractSchemaYAML")
def read_root(sql_config:SQLConfig,HF_KEY:str):
    try:
        logging.info("Starting the extraction of meta data from sql ")
        mysql_host=sql_config.MYSQL_HOST
        mysql_user=sql_config.MYSQL_USER
        mysql_password=sql_config.MYSQL_PASSWORD
        mysql_database=sql_config.MYSQL_DATABASE
        EAS=ExtractAndStoreMetaDataYAML(mysql_host,mysql_user,mysql_password,mysql_database,HF_KEY)

        logging.info("Extracting meta data from sql")
        # db_schema=EAS.extract_schema()
        # db_schema=EAS.extract_schema_new_format()
        db_schema=EAS.extract_schema_new_format_with_relation()
        logging.info("Formating the extracted schema")
        schema_texts=EAS.save_to_yaml(db_schema)
        logging.info("Finished storing meta data")
        print(schema_texts)
    except Exception as e:
        raise GraphDBProjectException(e,sys)
    
