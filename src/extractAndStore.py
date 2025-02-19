
from sqlalchemy import create_engine, inspect
import pandas as pd
import chromadb
from sqlalchemy import create_engine
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain.schema import Document
import yaml

class ExtractAndStoreMetaData():
    def __init__(self,mysql_host,mysql_user,mysql_password,mysql_database,HF_KEY):
        self.mysql_host=mysql_host
        self.mysql_user=mysql_user
        self.mysql_password=mysql_password
        self.mysql_database=mysql_database
        self.url=f"mysql+mysqlconnector://{self.mysql_user}:{self.mysql_password}@{self.mysql_host}:3306/{self.mysql_database}"
        self.HF_KEY=HF_KEY

    def extract_schema(self):# Create a connection to the MySQL database
        engine = create_engine(self.url)

        # Use SQLAlchemy Inspector to extract schema details
        inspector = inspect(engine)

        schema_info = {}

        for table_name in inspector.get_table_names():
            schema_info[table_name] = {
                "columns": [],
                "foreign_keys": []
            }

            # Get column details
            for column in inspector.get_columns(table_name):
                schema_info[table_name]["columns"].append({
                    "name": column["name"],
                    "type": str(column["type"])
                })

            # Get foreign key constraints
            for fk in inspector.get_foreign_keys(table_name):
                schema_info[table_name]["foreign_keys"].append({
                    "column": fk["constrained_columns"],
                    "references_table": fk["referred_table"],
                    "references_column": fk["referred_columns"]
                })

        engine.dispose()  # Close connection
        return schema_info
    
    def extract_schema_pandas(self):
        
        engine = create_engine(self.url)

        # Fetch all columns in one query
        query_columns = """
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = DATABASE();
        """
        df_columns = pd.read_sql(query_columns, engine)

        # Fetch all foreign key constraints in one query
        query_fks = """
        SELECT TABLE_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME 
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
        WHERE TABLE_SCHEMA = DATABASE() AND REFERENCED_TABLE_NAME IS NOT NULL;
        """
        df_fks = pd.read_sql(query_fks, engine)

        engine.dispose()  # Close connection

        # Convert to a structured dictionary for retrieval
        schema_dict = {}

        for table_name in df_columns["TABLE_NAME"].unique():
            schema_dict[table_name] = {
                "columns": df_columns[df_columns["TABLE_NAME"] == table_name][["COLUMN_NAME", "DATA_TYPE"]].to_dict(orient="records"),
                "foreign_keys": df_fks[df_fks["TABLE_NAME"] == table_name][["COLUMN_NAME", "REFERENCED_TABLE_NAME", "REFERENCED_COLUMN_NAME"]].to_dict(orient="records")
            }

        return schema_dict
    
    def format_schema_for_embedding(self,schema_dict):
        """Converts schema dictionary into a natural language text format."""
        schema_texts = []
        for table, details in schema_dict.items():
            text = f"Table: {table}\n"
            text += "Columns:\n"
            for col in details["columns"]:
                text += f" - {col['COLUMN_NAME']} ({col['DATA_TYPE']})\n"

            if details["foreign_keys"]:
                text += "Foreign Keys:\n"
                for fk in details["foreign_keys"]:
                    text += f" - {fk['COLUMN_NAME']} references {fk['REFERENCED_TABLE_NAME']}({fk['REFERENCED_COLUMN_NAME']})\n"

            schema_texts.append(text)
        return schema_texts
    
    # def store_schema_in_chromadb(self,schema_texts):
    #     os.environ["HF_KEY"]=self.HF_KEY
    #     model_name = "sentence-transformers/all-mpnet-base-v2"
    #     model_kwargs = {'device': 'cpu'}
    #     encode_kwargs = {'normalize_embeddings': False}
    #     embedding_model = HuggingFaceEmbeddings(
    #         model_name=model_name,
    #         model_kwargs=model_kwargs,
    #         encode_kwargs=encode_kwargs
    #     )

    #     """Stores schema embeddings in ChromaDB."""
    #     chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage

    #     # Create a collection for schema storage
    #     vector_db = Chroma(collection_name="mysql_schema", embedding_function=embedding_model, client=chroma_client)

    #     # Add schema documents with embeddings
    #     for i, text in enumerate(schema_texts):
    #         vector_db.add_texts(texts=[text], ids=[str(i)])

    #     print("✅ Schema stored in ChromaDB successfully!")

    def store_schema_in_chromadb(self,schema_texts):
        os.environ["HF_KEY"]=self.HF_KEY
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        # Assuming 'documents' and 'embeddings' are already defined
        documents = [Document(page_content=schema) for schema in schema_texts]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        # Define a persistent storage path
        persist_directory = "chroma_db"

        # Create the vectorstore with persistence
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_directory)

        # Save it to disk
        vectorstore.persist()
        print("✅ Schema stored in ChromaDB successfully!")


class ExtractAndStoreMetaDataYAML():
    def __init__(self,mysql_host,mysql_user,mysql_password,mysql_database,HF_KEY):
        self.mysql_host=mysql_host
        self.mysql_user=mysql_user
        self.mysql_password=mysql_password
        self.mysql_database=mysql_database
        self.url=f"mysql+mysqlconnector://{self.mysql_user}:{self.mysql_password}@{self.mysql_host}:3306/{self.mysql_database}"
        self.HF_KEY=HF_KEY

    def extract_schema(self):# Create a connection to the MySQL database
        engine = create_engine(self.url)
        inspector = inspect(engine)
        schema = {"tables": {}}
        # Get all tables
        tables = inspector.get_table_names()
        for table in tables:
            schema["tables"][table] = {"columns": [], "foreign_keys": []}
            # Get column details
            columns = inspector.get_columns(table)
            for col in columns:
                schema["tables"][table]["columns"].append({
                    "name": col["name"],
                    "type": str(col["type"]),
                    "primary_key": col["name"] in inspector.get_pk_constraint(table).get("constrained_columns", [])
                })
            # Get foreign keys
            fkeys = inspector.get_foreign_keys(table)
            for fk in fkeys:
                schema["tables"][table]["foreign_keys"].append({
                    "column": fk["constrained_columns"][0],
                    "references": f"{fk['referred_table']}.{fk['referred_columns'][0]}"
                })
        return schema
    

    def extract_schema_new_format(self, output_file="schema.yaml"):
        engine = create_engine(self.url)
        inspector = inspect(engine)
        schema_yaml = {}
        # Get all tables
        tables = inspector.get_table_names()
        for table in tables:
            columns = inspector.get_columns(table)
            pk_constraints = inspector.get_pk_constraint(table).get("constrained_columns", [])
            fkeys = inspector.get_foreign_keys(table)
            foreign_key_map = {}
            for fk in fkeys:
                foreign_key_map[fk["constrained_columns"][0]] = f"{fk['referred_table']}.{fk['referred_columns'][0]}"
            for col in columns:
                key = f"{table}.{col['name']}"
                schema_yaml[key] = [
                    {"datatype": str(col["type"])},
                ]
                if col["name"] in foreign_key_map:
                    schema_yaml[key].append({"foreign_key": [foreign_key_map[col["name"]]]})
            
        return schema_yaml
    
    def extract_schema_new_format_with_relation(self):
        engine = create_engine(self.url)
        inspector = inspect(engine)
        schema_yaml = {}
        table_relationships = {}  # To track relationships between tables

        # Get all tables
        tables = inspector.get_table_names()
        for table in tables:
            columns = inspector.get_columns(table)
            pk_constraints = inspector.get_pk_constraint(table).get("constrained_columns", [])
            fkeys = inspector.get_foreign_keys(table)
            foreign_key_map = {}

            # Process foreign keys
            for fk in fkeys:
                constrained_col = fk["constrained_columns"][0]
                referred_table = fk["referred_table"]
                referred_column = fk["referred_columns"][0]

                # Store the foreign key relationship
                foreign_key_map[constrained_col] = f"{referred_table}.{referred_column}"

                # Track relationships at the table level
                if table not in table_relationships:
                    table_relationships[table] = []
                table_relationships[table].append(f"REFERENCES {referred_table} ON {constrained_col} → {referred_column}")

            # Store column details
            for col in columns:
                key = f"{table}.{col['name']}"
                schema_yaml[key] = [{"datatype": str(col["type"])}]
                
                # If it's a foreign key, add reference information
                if col["name"] in foreign_key_map:
                    schema_yaml[key].append({"foreign_key": [foreign_key_map[col["name"]]]})

        # Merge relationships into schema output
        schema_yaml["_table_relationships_"] = table_relationships
        return schema_yaml
    
    def save_to_yaml(self, schema,filename="database_schema.yaml"):
        """Saves extracted schema to a YAML file."""
        with open(filename, "w") as file:
            yaml.dump(schema, file, default_flow_style=False)
        print(f"Schema successfully saved to {filename}")