from pydantic import BaseModel

class Neo4jConfig(BaseModel):
    NEO4J_URI:str
    NEO4J_USER:str
    NEO4J_PASSWORD:str

class SQLConfig(BaseModel):
    MYSQL_HOST:str
    MYSQL_USER:str
    MYSQL_PASSWORD:str
    MYSQL_DATABASE:str