from pydantic import BaseModel

class UserInput(BaseModel):
    user_query:str
    HF_KEY:str