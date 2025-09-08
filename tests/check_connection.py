from pymongo import MongoClient

client = MongoClient("mongodb://mongoadmin:secret123@localhost:27017/?authSource=admin")
db = client["voice_db"]

print(f"🎧 Medya kayıtları: {db['media_files'].count_documents({})}")
print(f"🗣️ Segment kayıtları: {db['segments'].count_documents({})}")
