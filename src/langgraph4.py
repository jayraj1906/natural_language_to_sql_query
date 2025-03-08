import yaml
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from typing import Annotated
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from my_sql_thing import extract_schema_from_mysql

def extract_yaml():
    with open('C:/Jayraj/Codes/yashwant sir/graphDBSQL/database_schema_old.yaml', 'r') as file:
        schema_dict = yaml.safe_load(file)
    return schema_dict


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


prompt="""
### Task
Generate a SQL query to answer [QUESTION]{user_question}[/QUESTION]

### Database Schema
The query will run on a database with the following schema:
{table_metadata_string_DDL_statements}

### Instruction
- DO NOT generate SQL queries with aggregations unless the user explicitly asks for them.  
- DO NOT make assumptions about missing or incomplete data.  
- DO NOT generate SQL queries with `REFERENCES` or foreign key constraints in the query itself.  
- DO NOT modify or create new table or column names. 
- Always prioritize simplicity while ensuring correctness.  

### Answer
Given the database schema, here is the SQL query that [QUESTION]{user_question}[/QUESTION]
[SQL]
"""


def generate_query(state:State):
    """Function that generates sql query"""
    user_query=state["messages"][0]
    table_metadata_ddl_statements=state["messages"][-1].content
    prompt_template = PromptTemplate.from_template(prompt)
    # query_gen_llm=ChatOllama(model="hf.co/defog/sqlcoder-7b-2",temperature=0)
    query_gen_llm=ChatOllama(model="codellama",temperature=0)
    pt=prompt_template.invoke({"user_question": user_query,"table_metadata_string_DDL_statements":table_metadata_ddl_statements})
    message=query_gen_llm.invoke(pt)
    return {"messages": [message]}

class YAMLToStr(BaseModel):
    """Yaml converted to table_metadata_string_DDL_statements"""
    yaml_to_str: str = Field(..., description="table_metadata_string_DDL_statements")


yaml_to_str_prompt_template = PromptTemplate(
    input_variables=["yaml_format_table_schema"],
    template="""
You are given the schema of multiple database tables, including their columns, data types, constraints, and relationships. Your task is to convert this schema into the corresponding table metadata DDL (Data Definition Language) statements.

***Expected Output Format (DDL Statements)***
- IMMEDIATELY OUTPUT ONLY THE DDL STATEMENTS
- DO NOT GENERATE ANY CODE TO DO THE TRANSFORMATION
- Do not include explanations, code, or extra text
- Preserve the exact table and column names as given in the schema
- Maintain case sensitivity as provided in the input
- Add primary keys and foreign key constraints where applicable

For each table:
- Define the table creation statement.
- Specify columns with their data types and constraints.
- Add primary keys and foreign key constraints where applicable.
- Ensure indexes are included where they exist.

If there is a relationship between the tables, such as a foreign key constraint, you should include the foreign key constraint in the appropriate table's DDL.

***INPUT SCHEMA***
{yaml_format_table_schema}
---

**Example Input:**

<EXTRACTEDSCHEMA>
{{'TABLE_NAME': 'table1'}}
{{'TABLE_NAME': 'table2'}}
{{'TABLE_NAME': 'table3'}}
Table: table1
Columns:
  - column1 (data_type) NOT NULL
  - column2 (data_type) NOT NULL DEFAULT some_value
Primary Key(s):
  - column1
Foreign Key(s):
  - fk_name: column2 -> table2.columnA
Index(es):
  - column2 on column2
  - UNIQUE PRIMARY on column1
...
<EXTRACTEDSCHEMA>

**Example Output (DDL Statements):**

```sql
CREATE TABLE table1 (
    column1 data_type NOT NULL, -- Description of column1
    column2 data_type NOT NULL DEFAULT some_value, -- Description of column2
    PRIMARY KEY (column1), -- Primary key constraint on column1
    FOREIGN KEY (column2) REFERENCES table2(columnA), -- Foreign key referencing table2
    INDEX (column2) -- Index on column2
);

CREATE TABLE table2 (
    columnA data_type NOT NULL DEFAULT some_value, -- Description of columnA
    columnB data_type NOT NULL, -- Description of columnB
    PRIMARY KEY (columnA), -- Primary key constraint on columnA
);

CREATE TABLE table3 (
    columnX data_type NOT NULL, -- Description of columnX
    columnY data_type, -- Description of columnY
    PRIMARY KEY (columnX), -- Primary key constraint on columnX
    FOREIGN KEY (columnY) REFERENCES table1(column2), -- Foreign key referencing table1
);
```
    """)

# yaml_to_str_llm = ChatOllama(model="llama3.2",temperature=0).bind_tools([YAMLToStr])
yaml_to_str_llm = ChatOllama(model="llama3.2",temperature=0)

def yaml_to_str_convert(state: State):
    """Function that formats the YAML schema into DDL statements"""
    HOST=""
    USER=""
    PASSWORD=""
    DATABASE=""
    # yaml_schema=state["messages"][-1].tool_calls[0]["args"]["fields"]
    database_schema=extract_schema_from_mysql(HOST,USER,PASSWORD,DATABASE)
    prompt_template_format=yaml_to_str_prompt_template.format(yaml_format_table_schema=database_schema)
    message=yaml_to_str_llm.invoke(prompt_template_format)
    return  {"messages": [message]}

def first_tool_call(state: State):
    yaml_schema=extract_yaml()
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "extract_relevant_columns_tool",
                        "args": {
                            "fields":yaml_schema
                        },
                        "id": "parsed_yaml_Extracted",
                    }
                ],
            )
        ]
    }



workflow = StateGraph(State)
# workflow.add_node("first_tool_call", first_tool_call)
workflow.add_node("yaml_to_str_convert", yaml_to_str_convert)
workflow.add_node("generate_query", generate_query)



workflow.add_edge(START,"yaml_to_str_convert")
workflow.add_edge("yaml_to_str_convert","generate_query")
workflow.add_edge("generate_query",END)

############################## Edge Creation #####################################

app = workflow.compile()
# user_query="Get me the list of top 10 populated cities"
# user_query="Filters cities where their population is greater than the average city population."
user_query="find countries where more than 50% of the population speaks an official language."
# messages=app.invoke({"messages": [("user", user_query)]},{"recursion_limit": 100})

for events in app.stream({"messages": [("user", user_query)]}):
    print(events)