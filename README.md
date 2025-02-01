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

- Go to Groq API and sign up or log in.
- Navigate to the API section and generate your API key.
- Store the key in a secure location (you will need it later).

### 5. Create a .env file 
Create a ".env" file and store the api keys in that file

```
GROQ_API_KEY=your_api_key_here
```

### 6. Run the Application
Finally, run the Streamlit application using the following command:

```bash
streamlit run app.py
```
Your app should now be up and running. Happy coding!