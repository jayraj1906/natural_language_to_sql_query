from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
import os

load_dotenv()
database_link=os.getenv("DB")
db=SQLDatabase.from_uri(database_link)
llm = ChatOllama(model = "llama3.2")
agent_executor = create_sql_agent(llm, db = db, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True)
agent_executor.invoke("Calculates GDP per capita (GNP ÷ Population) and finds the country with the highest value")

