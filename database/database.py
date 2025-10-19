from pymongo.mongo_client import MongoClient


class Database:
    def __init__(self, uri: str, db_name: str):
        self.client = MongoClient(uri)

        try:
            self.client.admin.command("ping")
            print("Database connected")
        except Exception as e:
            print(e)

        self.db = self.client[db_name]

    def get_collection(self, collection_name: str):
        return self.db[collection_name]

    def insert_many(self, collection_name: str, documents: list):
        collection = self.get_collection(collection_name)
        result = collection.insert_many(documents)
        return result.inserted_ids

    def find_documents(self, collection_name: str, query: dict):
        collection = self.get_collection(collection_name)
        return list(collection.find(query))
    
    def count_documents(self, collection_name: str, query: dict):
        collection = self.get_collection(collection_name)
        return collection.count_documents(query)


def get_database(uri: str, db_name: str) -> Database:
    return Database(uri=uri, db_name=db_name)