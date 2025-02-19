from fastapi import FastAPI
from src.neo_graph_db import ExtractAndStoreInGraph
from fastapi import HTTPException
from src.my_logging.logger import logging
from src.exception.exception import GraphDBProjectException
from src.schema.sql_and_neodb import SQLConfig,Neo4jConfig
from src.schema.getResponseSchema import UserInput
from src.extractAndStore import ExtractAndStoreMetaData,ExtractAndStoreMetaDataYAML
from src.semantic_search import generate_sql_query,recheck_sql_query
import sys
import json
from src.getResponse import GenerateResponse
import yaml

app=FastAPI()


@app.post("/graphdb/extract_and_store")
def read_root(sql_config:SQLConfig,neo_graph_db_config:Neo4jConfig):
    try:
        logging.info("Starting the extraction of meta data from sql and storing it in Neo4j Graph DB")
        neo4j_uri=neo_graph_db_config.NEO4J_URI
        neo4j_user=neo_graph_db_config.NEO4J_USER
        neo4j_password=neo_graph_db_config.NEO4J_PASSWORD
        mysql_host=sql_config.MYSQL_HOST
        mysql_user=sql_config.MYSQL_USER
        mysql_password=sql_config.MYSQL_PASSWORD
        mysql_database=sql_config.MYSQL_DATABASE
        ESG=ExtractAndStoreInGraph(neo4j_uri,neo4j_user,neo4j_password,mysql_host,mysql_user,mysql_password,mysql_database)
        logging.info("Extracting meta data from sql")
        db_schema=ESG.extract_schema()
        logging.info("Extracting of meta data from sql finished, Now starting to store the meta data into Neoj Graph db")
        final=ESG.store_schema_in_neo4j(db_schema)
        logging.info("Finished storing meta data into Neoj Graph db")
        print(final)
    except Exception as e:
        raise GraphDBProjectException(e,sys)

@app.post("/rag/extractSchema")
def read_root(sql_config:SQLConfig,HF_KEY:str):
    try:
        logging.info("Starting the extraction of meta data from sql ")
        mysql_host=sql_config.MYSQL_HOST
        mysql_user=sql_config.MYSQL_USER
        mysql_password=sql_config.MYSQL_PASSWORD
        mysql_database=sql_config.MYSQL_DATABASE
        EAS=ExtractAndStoreMetaData(mysql_host,mysql_user,mysql_password,mysql_database,HF_KEY)

        logging.info("Extracting meta data from sql")
        db_schema=EAS.extract_schema_pandas()
        logging.info("Formating the extracted schema")
        schema_texts=EAS.format_schema_for_embedding(db_schema)
        logging.info("Storing the formatted schema in vector db")
        schema_store=EAS.store_schema_in_chromadb(schema_texts)
        logging.info("Finished storing meta data")
        print(schema_store)
    except Exception as e:
        raise GraphDBProjectException(e,sys)
    
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
    
@app.post("/getresponse")
def read_root(user_input:UserInput):
    try:
        response=GenerateResponse(user_input.user_query,user_input.HF_KEY)
        result=response.generate_response()
        print(result)
    except Exception as e:
        raise GraphDBProjectException(e,sys)
    
@app.post("/query")
def generate_sql_from_nlp(user_query: str,filename="database_schema.yaml"):
    with open(filename, "r") as file:
        schema=yaml.safe_load(file)
    sql_query = generate_sql_query(user_query, schema)
    checked_sql_query=recheck_sql_query(sql_query,user_query,schema)
    return {"sql_query": checked_sql_query}
