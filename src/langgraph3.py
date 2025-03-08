from typing import Any
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from langchain_core.tools import tool
from typing import Annotated
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from langchain_ollama import ChatOllama
import yaml
from typing import Dict,List
from langchain.schema import Document
from rank_bm25 import BM25Okapi 
import spacy

load_dotenv()
nlp = spacy.load("en_core_web_sm")  # Load a suitable spaCy model
os.environ["HF_KEY"]=os.getenv("HF_TOKEN")

with open('C:/Jayraj/Codes/yashwant sir/graphDBSQL/database_schema.yaml', 'r') as file:
    schema_dict = yaml.safe_load(file)


model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
schema_keys = list(schema_dict.keys())
documents = [Document(page_content=key) for key in schema_keys]
# tokenized_docs = [key.lower().split() for key in schema_keys]
tokenized_docs = [key.lower().split('.')[-1].split() for key in schema_keys] 
bm25 = BM25Okapi(tokenized_docs)

# Initialize FAISS vector store with OpenAI embeddings
vectorstore = FAISS.from_documents(documents, embeddings)
BM25_THRESHOLD = 2  # Minimum BM25 score required for a match
FAISS_THRESHOLD = 0.85  # Cosine similarity (normalized)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def find_best_matches(extracted_fields):
    """
    Find similar column and table names using hybrid search (vector + keyword)
    """
    results = {}
    for field in extracted_fields:
        best_matches = hybrid_search(field, k=1)  # Perform hybrid search
        if best_matches:
            best_match = best_matches[0]  # Take the best result
            results[field] = f"{best_match}: {schema_dict.get(best_match, 'UNKNOWN')}"
    
    return results if results else {} 

def hybrid_search(query, k=1):
    """Performs hybrid search, prioritizing BM25, with FAISS as backup."""
    query_lower = query.lower()

    # 1️⃣ BM25 Keyword Search
    bm25_scores = bm25.get_scores(query_lower.split())
    sorted_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    
    keyword_matches = []
    for i in sorted_indices:
        if bm25_scores[i] >= BM25_THRESHOLD:
            keyword_matches.append(schema_keys[i])
    
    # 2️⃣ FAISS Vector Search (Only if BM25 fails)
    faiss_results = []
    if not keyword_matches:  
        faiss_results = vectorstore.similarity_search_with_score(query, k=k)
        faiss_matches = [doc.page_content for doc, score in faiss_results if score >= FAISS_THRESHOLD]
    else:
        faiss_matches = []  # Ignore FAISS if BM25 is strong

    # 3️⃣ Combine Results (Prioritize BM25, then FAISS)
    hybrid_results = keyword_matches or faiss_matches  

    return hybrid_results[:k]  # Return only the best match

@tool
def extract_keywords(text):
    """
    Takes in a user input in natural language
    preprocess it to get only the important columns
    """
    doc = nlp(text)
    keywords = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN", "ADJ"]:  # Extract nouns, proper nouns, and adjectives
            keywords.append(token.lemma_.lower()) #Lemmatization and lowercasing for consistency

    #Remove stop words and punctuations
    keywords = [token.text for token in doc if not token.is_stop and not token.is_punct and token.text in keywords]

    #Remove duplicates while preserving order (using set is not appropriate as it disregards order)
    unique_keywords = []
    seen = set()
    for item in keywords:
        if item not in seen:
            unique_keywords.append(item)
            seen.add(item)
    return unique_keywords

def first_tool_call(state: State):
    
    fields_ext = state["messages"][-1].tool_calls[0]["args"]["fields"]
    tokenized_words = []
    for entity in fields_ext:
        doc = nlp(entity)
        tokenized_words.extend([token.text.lower() for token in doc]) 
    extracted_columns_from_yaml = find_best_matches(tokenized_words)
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "extract_relevant_columns_tool",
                        "args": {
                            "fields":extracted_columns_from_yaml
                        },
                        "id": "parsed_yaml_Extracted",
                    }
                ],
            )
        ]
    }


class SqlQueryGenerated(BaseModel):
    """Missing columns are relevant or not"""
    sql_query: str = Field(..., description="Generated sql query" )
sql_query_generate_prompt_template = PromptTemplate(
    input_variables=["user_query","schema_fields","table_relationship"],
    template="""
**Role & Task:**  
You are an AI that generates SQL queries based on:  
1. **User query** (natural language request) {user_query}  
2. **Schema fields** (mapping of user-friendly terms to actual database tables and columns) {schema_fields}
3. **Table relationships** (explicit foreign key relationships between tables)  {table_relationship}

Your goal is to generate **syntactically correct and logically structured** SQL queries **without making any assumptions** beyond the provided schema and relationships.  

### **Input Variables**  
- **`schema_fields`**: A dictionary mapping user-friendly terms to database columns in the format:  
- **`table_relationships`**: A dictionary defining relationships between tables using explicit foreign key references:  
- **`user_query`**: A natural language request.


### **Query Construction Guidelines**  
1. **Column & Table Mapping**  
   - Translate user query terms using `schema_fields`.  
   - Identify which **tables** contain the required columns.  

2. **Joining Tables Using `table_relationships`**  
   - If multiple tables are involved, determine how they are connected using `table_relationships`.  
   - Use **JOINs only when necessary** based on explicit foreign key relationships.  
   - If no relationship exists, do **not** force a join.  

3. **Strict Adherence to Schema**  
   - Do **not** infer missing information.  
   - Do **not** add extra aggregations (`SUM`, `AVG`, etc.) unless explicitly requested.  
   - Keep queries **as simple as possible** while ensuring correctness.  
---

### **Final Instructions for the LLM**  
- **DO NOT** make assumptions beyond the provided schema and relationships.  
- **DO NOT** generate SQL queries with joins unless explicitly required by `table_relationships`.  
- **DO NOT** introduce aggregations unless the user explicitly asks for them.  
- **ALWAYS** prioritize simplicity while maintaining logical correctness.  

"""
)

# """
# You are an AI that generates SQL queries based on a user query in natural language and a structured schema mapping. Your task is to output a syntactically correct and logically structured SQL query.  
# user_query:{user_query}
# schema_fields:{schema_fields}
# ### **Guidelines:**  
# 1. **Schema Mapping:** You will receive a dictionary where:  
#     - The **keys** represent user-friendly field names.  
#     - The **values** specify the corresponding SQL table and column in the format

# 2. **SQL Query Construction:**  
#    - Translate user terms in the query into their corresponding table columns.  
#    - Construct a **clear, logically structured SQL query**.  
#    - Use **JOINs only if necessary** to combine data from multiple tables.  
#    - Do **not hallucinate** or introduce columns, tables, or conditions not present in the schema mapping.  
#    - **Keep it simple**: Avoid unnecessary aggregation (`SUM`, `AVG`, etc.) unless explicitly asked.  
# ### **Final Instructions:**  
# - Always ensure that the SQL query is syntactically correct and follows best practices.  
# - Stick to the provided schema and **do not infer** missing information.  
# - Return **only the SQL query** without explanations unless explicitly requested.
#     """



# query_gen = ChatOllama(model="codellama",temperature=0)
query_gen = ChatOllama(model="llama3.2",temperature=0).bind_tools([SqlQueryGenerated])
def query_gen_node(state: State):
    with open('C:/Jayraj/Codes/yashwant sir/graphDBSQL/database_schema.yaml', 'r') as file:
        table_relation = yaml.safe_load(file)["_table_relationships_"]
    user_query=state["messages"][0].content
    fields=state["messages"][-1].tool_calls[0]["args"]["fields"]
    message = query_gen.invoke(sql_query_generate_prompt_template.format(schema_fields=fields,user_query=user_query,table_relationship=table_relation))
    return {"messages": [message]}


class ExtractedEntities(BaseModel):
    """Missing columns are relevant or not"""
    entities: List[str] = Field(..., description="Extracted entities" )
entities_extract_prompt_template = PromptTemplate(
    input_variables=["query"],
    template="""
    Given the following user query: "{query}", identify and extract the key information types the user wants.
    List the extracted entities as a Python list.
    Example:
    User Query: "Get me the list of top 10 most populated cities."
    Extracted Entities: ["population", "city name"]
    Now, extract entities for this query: "{query}"
    Extracted Entities:
    """
)
entities_extract_column = ChatOllama(model="llama3.2", temperature=0).bind_tools([ExtractedEntities])


def first_tool_call_spacy(state: State):
    userQuery=state["messages"][-1].content
    extracted_entities=entities_extract_column.invoke(entities_extract_prompt_template.format(query=userQuery))
    final_extracted_entities=extracted_entities.tool_calls[0]["args"]["entities"]
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "extract_entity_from_user_query",
                        "args": {
                            "fields":final_extracted_entities
                        },
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }


workflow = StateGraph(State)
workflow.add_node("spacyTool", first_tool_call_spacy)
workflow.add_node("first_tool_call", first_tool_call)
workflow.add_node("query_making_sense_or_not", first_tool_call)
workflow.add_node("gen_query",query_gen_node)

workflow.add_edge(START, "spacyTool")
workflow.add_edge("spacyTool", "first_tool_call")
workflow.add_edge("first_tool_call","gen_query")
workflow.add_edge("gen_query",END)

############################## Edge Creation #####################################

app = workflow.compile()
# user_query="Get me the list of top 10 populated cities"
user_query="Filters cities where their population is greater than the average city population."
# user_query="find countries where more than 50% of the population speaks an official language."
# messages=app.invoke({"messages": [("user", user_query)]},{"recursion_limit": 100})

for events in app.stream({"messages": [("user", user_query)]}):
    print(events)