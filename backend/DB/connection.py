from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

MONGO_URI = "mongodb://localhost:27017"

try:
    client = MongoClient(MONGO_URI)
    db = client["reconciliation_db"]
    print("Mongo DB connected successfully")
except ConnectionFailure as e:
    print("Mongo DB connection failed...",e)
    db = none