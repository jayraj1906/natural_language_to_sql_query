# Natural langauge query to SQL query

## Setting up the Environment

Follow the steps below to set up the project and run the application.

### 1. Create a Virtual Environment

First, you'll need to create a virtual environment. Follow the platform-specific instructions:


```
python -m venv venv 

python3 -m venv venv
```

### 2. Activate the Virtual Environment
After creating the virtual environment, activate it using the command below based on your platform:

On Windows:
```bash
.\venv\Scripts\activate
```
On macOS/Linux:
```bash
source venv/bin/activate
```

### 3. Install Dependencies 
Next, install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### 4. Get the Groq API Key
You need a Groq API key to run the application. Follow these steps:

- Go to [Groq API](https://console.groq.com/) and sign up or log in.
- Navigate to the API section and generate your API key.
- Store the key in a secure location (you will need it later).

### 5. Create a .env file 
Create a ".env" file and store the api keys in that file

```
GROQ_API_KEY="your_api_key_here"
```
also store the link of your mysql database as given below in you .env file

```
DB="mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
```
if you have special charaters in your password make sure to encode it ex: @ -> %40

### 6. Run the Application
Finally, run the Streamlit application using the following command:

```bash
python natural_language_to_sql_query.py
```

### In this project two llm models are used: 
-  "deepseek-r1-distill-llama-70b" : Because it is trending right now but this llm does not support function calling and that is why i have to use another large language model to which support function calling - "llama-3.3-70b-versatile"

- If you want to change the llm model and try 
- open "natural_language_to_sql_query.py" file and change the llm model stored in variable name "llm_model" and "llm_model_2" in line 32 and 33
```
llm_model = "deepseek-r1-distill-llama-70b"
llm_model_2="llama-3.3-70b-versatile"
```
Note : Groq have different token limits per second for different model

Your app should now be up and running. Happy coding!