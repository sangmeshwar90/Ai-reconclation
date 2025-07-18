# routes/auth.py

from fastapi import APIRouter, HTTPException
from models.user_model import UserSignup, UserLogin
from DB.connection import db

router = APIRouter()
users_collection = db["users"]

@router.post("/signup")
def signup(user: UserSignup):
    existing = users_collection.find_one({"email": user.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_data = {
        "name": user.name,
        "email": user.email,
        "password": user.password  # storing plain text password (⚠ not secure)
    }
    users_collection.insert_one(user_data)
    return {"message": "User signed up successfully "}

@router.post("/login")
def login(user: UserLogin):
    existing = users_collection.find_one({"email": user.email})
    if not existing or user.password != existing["password"]:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    return {"message": f"Welcome, {existing['name']}!"}

@router.get("/hello")
def hello():
    return {"message": "Hello from backend"}