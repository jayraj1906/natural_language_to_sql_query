from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated, Literal

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages


load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
db=SQLDatabase.from_uri("mysql+mysqlconnector://root:Jayraj%401906@localhost:3306/netflix_data")

llm_model = "deepseek-r1-distill-llama-70b"
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


toolkit = SQLDatabaseToolkit(db=db, llm=ChatGroq(model=llm_model))
tools = toolkit.get_tools()
############################## Tools #####################################
#Tool number 1
list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")

#Tool number 2
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

#Tool number 3
@tool
def db_query_tool(query: str) -> str:
    """
    Execute a SQL query against the database and get back the result.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    result = db.run_no_throw(query)
    if not result:
        return "Error: Query failed. Please rewrite your query and try again."
    return result

############################## Tools #####################################

############################## LLM Will check if the generated query is correct or not #####################################
query_check_system="""You are a SQL expert with a strong attention to detail.
Double-check the SQLite query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, reproduce the original query.

### IMPORTANT ###
Always call the `db_query_tool` function to execute the query after checking it.

"""


query_check_prompt = ChatPromptTemplate.from_messages(
    [("system", query_check_system), ("placeholder", "{messages}")]
)
query_check = query_check_prompt | ChatGroq(model="llama-3.3-70b-versatile", temperature=0).bind_tools(
    [db_query_tool], tool_choice="required"
)
############################## LLM Will check if the generated query is correct or not #####################################


############################## LLM Query generation #####################################
# Describe a tool to represent the end state
class SubmitFinalAnswer(BaseModel):
    """Submit the final answer to the user based on the query results."""

    final_answer: str = Field(..., description="The final answer to the user")

query_gen_system="""
You are a SQL expert who only outputs valid, syntactically correct SQLite queries. 

Given an input question, strictly output **only** the SQL query—nothing else.

- Do **not** provide explanations, context, or reasoning.
- Do **not** include natural language text before or after the query.
- Do **not** submit a final answer—just return the SQL query.
- **Only output the SQL query as raw text, enclosed in triple backticks (` ```sql ... ``` `).**
  
Workflow behavior:
1. The first time you generate a query, it will be checked for correctness before execution.
2. If execution fails (returns an error), you must generate a new query to retry.
3. **Only after a successful query execution, the final answer will be submitted.**

If you get an error while executing a query, **rewrite the query and try again**.

If you get an empty result set, you should try to rewrite the query to get a non-empty result set.  
NEVER make stuff up if you don't have enough information to answer the query—just say you don't have enough information.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP, etc.) to the database.

Example:
If the question is: "Find the total number of payments in 2023."
You **must** output:
```sql
SELECT COUNT(*) FROM payments WHERE YEAR(paymentDate) = 2023;
Nothing more, nothing less.

"""

query_gen_prompt = ChatPromptTemplate.from_messages(
    [("system", query_gen_system), ("placeholder", "{messages}")]
)
query_gen = query_gen_prompt | ChatGroq(model="llama-3.3-70b-versatile", temperature=0).bind_tools(
    [SubmitFinalAnswer]
)

############################## LLM Query generation #####################################


############################## check_for_missing_system_prompt #####################################

check_for_missing_system_prompt="""You are an intelligent SQL assistant responsible for verifying database schemas to ensure all necessary columns exist before generating queries.  

Your task is to check if the retrieved schema contains all the columns needed to answer the user’s query. If any column is missing, search other tables for it.  

**⚠️ Important:**  
🔹 **Do NOT solve or answer the user’s question.**  
🔹 **Your only task is to verify whether all required columns exist and determine where to retrieve them from.**  

### **Instructions:**  

1. **Verify Retrieved Schema:**  
   - Examine the schema retrieved using the `get_schema_tool`.  
   - Compare it with the columns required to answer the user’s query.  
   - Ensure **all** requested columns are present in the retrieved schema.  

2. **Handle Missing Columns:**  
   - If a required column **is missing** in the retrieved schema, return an error:  
     **"Error: Column '<column_name>' not found in '<table_name>'. Searching other tables for this column."**  
   - Check the schemas of other tables using the `get_schema_tool` to find the missing column.  

3. **Find Table Relationships:**  
   - If the missing column is found in another table, determine how it relates to the initially selected table (e.g., through foreign keys or common identifiers).  
   - Identify if a **JOIN operation** is needed and specify the type of join required (INNER JOIN, LEFT JOIN, etc.).  

4. **Final Decision:**  
   - If all required columns are found across multiple tables, return the updated schema details along with table relationships.  
   - If a required column **is not found in any table**, return an error:  
     **"Error: The required column '<column_name>' does not exist in any table. Please check the database structure."**  

🚫 **Do NOT attempt to generate SQL queries or answer the user’s question.** Your role is strictly to verify column availability and identify table relationships.

"""

check_missing_prompt = ChatPromptTemplate.from_messages(
    [("system", check_for_missing_system_prompt), ("placeholder", "{messages}")]
)
missing_column = check_missing_prompt | ChatGroq(model=llm_model, temperature=0)

############################## check_for_missing_system_prompt #####################################
############################## Node function calling #####################################
# Add a node for the first tool call
def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }

def model_check_query(state: State) -> dict[str, list[AIMessage]]:
    """
    Use this tool to double-check if your query is correct before executing it.
    """
    return {"messages": [query_check.invoke({"messages": [state["messages"][-1]]})]}

def query_gen_node(state: State):
    message = query_gen.invoke(state)
    # Sometimes, the LLM will hallucinate and call the wrong tool. We need to catch this and return an error message.
    tool_messages = []
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] != "SubmitFinalAnswer":
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                        tool_call_id=tc["id"],
                    )
                )
    else:
        tool_messages = []
    return {"messages": [message] + tool_messages}

# Define a conditional edge to decide whether to continue or end the workflow
def should_continue(state: State) -> Literal[END, "correct_query", "query_gen"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there is a tool call, then we finish
    if getattr(last_message, "tool_calls", None):
        return END
    if last_message.content.startswith("Error:"):
        return "query_gen"
    else:
        return "correct_query"
    
def missing_or_not(state: State) -> Literal["query_gen", "model_get_schema"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.content.startswith("Error:"):
        return "model_get_schema"
    else:
        return "query_gen"
    

############################## Node function calling #####################################

    

############################## Choosing relevant tables for schemas #####################################

choosing_table_system="""
You are a SQL assistant responsible for retrieving table schemas to help generate SQL queries.  
Your task is to determine which tables are needed based on the user’s question **without making assumptions** about column existence.  

### **Instructions:**
1. **Identify Relevant Tables:**  
   - Look at all available tables in the database.  
   - Select tables that are likely to contain the required columns.  

2. **Check for Column Existence:**  
   - Do not assume all required columns exist in the first table you choose.  
   - Retrieve the schema of each selected table using the `get_schema_tool`.  
   - **Verify** whether the required columns exist in the schema response.  

3. **Handle Missing Columns:**  
   - If a required column is **not found** in the retrieved schema, return an error:  
     **"Error: Column '<column_name>' not found in '<table_name>'. Check other tables for this column."**  
   - Continue searching other tables to locate the missing column.  
   - If the column is found in another table, determine how the tables are related (e.g., using foreign keys) to join them correctly.  

4. **Final Decision:**  
   - If all required columns are found, proceed with query generation.  
   - If any required column is **not found in any table**, return a final error:  
     **"Error: The required column '<column_name>' was not found in any table. Please check the database structure."**  
""" 
model_get_schema_prompt = ChatPromptTemplate.from_messages(
    [("system", choosing_table_system), ("placeholder", "{messages}")]
)

# Add a node for a model to choose the relevant tables based on the question and available tables
model_get_schema = model_get_schema_prompt | ChatGroq(model=llm_model, temperature=0).bind_tools([get_schema_tool])

# model_get_schema = ChatGroq(model=llm_model, temperature=0).bind_tools([get_schema_tool])

############################## Choosing relevant tables for schemas #####################################



############################## Node Creation #####################################

workflow = StateGraph(State)
workflow.add_node("first_tool_call", first_tool_call)

# Add nodes for the first two tools
workflow.add_node(
    "list_tables_tool", create_tool_node_with_fallback([list_tables_tool])
)
workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))
workflow.add_node(
    "model_get_schema",
    lambda state: {
        "messages": [model_get_schema.invoke({"messages":state["messages"]})],
    },
)

workflow.add_node("query_gen", query_gen_node)

# Add a node for the model to check the query before executing it
workflow.add_node("correct_query", model_check_query)
workflow.add_node("check_for_required_columns", 
                  lambda state: {
        "messages": [missing_column.invoke({"messages":state["messages"]})],
    },)

# Add node for executing the query
workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))

############################## Node Creation #####################################

############################## Edge Creation #####################################

# Specify the edges between the nodes
workflow.add_edge(START, "first_tool_call")
workflow.add_edge("first_tool_call", "list_tables_tool")
workflow.add_edge("list_tables_tool", "model_get_schema")
workflow.add_edge("model_get_schema", "get_schema_tool")
workflow.add_edge("get_schema_tool","check_for_required_columns")
workflow.add_conditional_edges("check_for_required_columns",missing_or_not)

workflow.add_conditional_edges(
    "query_gen",
    should_continue,
)
# workflow.add_edge( "query_gen","check_for_required_columns")
workflow.add_edge("correct_query", "execute_query")
workflow.add_edge("execute_query", "query_gen")

############################## Edge Creation #####################################

app = workflow.compile()
user_query="What are the weekly top-ranked English-language movies as of December 9, 2024, including their view rank, title, hours viewed, runtime, total views, and cumulative weeks in the top 10, ordered by view rank in ascending order?"
# messages=app.invoke({"messages": [("user", user_query)]},{"recursion_limit": 100})

# json_str = messages["messages"][-1].tool_calls[0]["args"]["final_answer"]
# print(json_str)
for events in app.stream({"messages": [("user", user_query)]}):
    print(events)