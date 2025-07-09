from fastapi import FastAPI
from DB.connection import db

app = FastAPI()

@app.get('/')
def read_root():
    if db:
        return{"message": "Hello from backend and mongoDb connected"}
    else:
        return{"Message": "Mongo connection error"}
         


@app.get('/profile')
def read_root():
    return{"welcome to the profile route..."}