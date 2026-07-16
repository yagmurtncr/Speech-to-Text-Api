import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
uri = os.getenv("MONGO_URI")

try:
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    print("✅ Bağlantı başarılı:", client.server_info()["version"])
except Exception as e:
    print("❌ Bağlantı hatası:", e)
