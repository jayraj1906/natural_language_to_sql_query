import mysql.connector
import json

# Connect to the MySQL database
def connect_to_mysql(host, user, password, database):
    connection = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    return connection

# Extract schema information
def extract_schema(connection):
    cursor = connection.cursor(dictionary=True)
    
    # Query tables in the database
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = %s", (connection.database,))
    tables = cursor.fetchall()

    schema = {}
    
    for table in tables:
        table_name = table['TABLE_NAME']
        schema[table_name] = {}
        
        # Query columns for each table
        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT, CHARACTER_MAXIMUM_LENGTH
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
        """, (connection.database, table_name))
        columns = cursor.fetchall()
        
        schema[table_name]['columns'] = []
        for column in columns:
            column_info = {
                'column_name': column['COLUMN_NAME'],
                'data_type': column['DATA_TYPE'],
                'is_nullable': column['IS_NULLABLE'],
                'default': column['COLUMN_DEFAULT'],
                'max_length': column['CHARACTER_MAXIMUM_LENGTH']
            }
            schema[table_name]['columns'].append(column_info)
        
        # Query indexes for each table
        cursor.execute("""
            SELECT INDEX_NAME, COLUMN_NAME, NON_UNIQUE
            FROM information_schema.statistics
            WHERE table_schema = %s AND table_name = %s
        """, (connection.database, table_name))
        indexes = cursor.fetchall()
        
        schema[table_name]['indexes'] = []
        for index in indexes:
            index_info = {
                'index_name': index['INDEX_NAME'],
                'column_name': index['COLUMN_NAME'],
                'is_unique': not index['NON_UNIQUE']
            }
            schema[table_name]['indexes'].append(index_info)
        
        # Query primary keys for each table
        cursor.execute("""
            SELECT COLUMN_NAME
            FROM information_schema.key_column_usage
            WHERE table_schema = %s AND table_name = %s AND CONSTRAINT_NAME = 'PRIMARY'
        """, (connection.database, table_name))
        primary_keys = cursor.fetchall()
        
        schema[table_name]['primary_keys'] = [pk['COLUMN_NAME'] for pk in primary_keys]
        
        # Query foreign keys for each table
        cursor.execute("""
            SELECT CONSTRAINT_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
            FROM information_schema.key_column_usage
            WHERE table_schema = %s AND table_name = %s AND REFERENCED_TABLE_NAME IS NOT NULL
        """, (connection.database, table_name))
        foreign_keys = cursor.fetchall()
        
        schema[table_name]['foreign_keys'] = []
        for fk in foreign_keys:
            foreign_key_info = {
                'constraint_name': fk['CONSTRAINT_NAME'],
                'column_name': fk['COLUMN_NAME'],
                'referenced_table': fk['REFERENCED_TABLE_NAME'],
                'referenced_column': fk['REFERENCED_COLUMN_NAME']
            }
            schema[table_name]['foreign_keys'].append(foreign_key_info)
    
    cursor.close()
    return schema

# Format the schema to a human-readable format
def format_schema_for_llm(schema):
    formatted_schema = ""
    
    for table, table_data in schema.items():
        formatted_schema += f"Table: {table}\n"
        
        formatted_schema += "Columns:\n"
        for column in table_data['columns']:
            column_info = f"  - {column['column_name']} ({column['data_type']})"
            if column['is_nullable'] == 'NO':
                column_info += " NOT NULL"
            if column['default'] is not None:
                column_info += f" DEFAULT {column['default']}"
            if column['max_length']:
                column_info += f" Max Length: {column['max_length']}"
            formatted_schema += f"{column_info}\n"
        
        if table_data['primary_keys']:
            formatted_schema += "Primary Key(s):\n"
            formatted_schema += f"  - {', '.join(table_data['primary_keys'])}\n"
        
        if table_data['foreign_keys']:
            formatted_schema += "Foreign Key(s):\n"
            for fk in table_data['foreign_keys']:
                formatted_schema += (f"  - {fk['constraint_name']}: "
                                      f"{fk['column_name']} -> {fk['referenced_table']}.{fk['referenced_column']}\n")
        
        if table_data['indexes']:
            formatted_schema += "Index(es):\n"
            for index in table_data['indexes']:
                unique_str = "UNIQUE " if index['is_unique'] else ""
                formatted_schema += (f"  - {unique_str}{index['index_name']} on {index['column_name']}\n")
        
        formatted_schema += "\n"
    
    return formatted_schema

# Main function to extract and format the schema
def extract_schema_from_mysql(HOST,USER,PASSWORD,DATABASE):
    # Set up connection details (modify as needed)
    host = HOST
    user = USER
    password = PASSWORD
    database = DATABASE
    
    connection = connect_to_mysql(host, user, password, database)
    
    try:
        schema = extract_schema(connection)
        formatted_schema = format_schema_for_llm(schema)
        # print(formatted_schema)
        return formatted_schema
    finally:
        connection.close()

if __name__ == "__main__":
    HOST="localhost"
    USER="root"
    PASSWORD="Jayraj@1906"
    DATABASE="world"
    extract_schema_from_mysql(HOST,USER,PASSWORD,DATABASE)