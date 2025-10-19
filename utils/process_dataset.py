import os
import pandas as pd
from database.database import get_database


def process_dataset():
    dataset_path = "data/heart.csv"
    dataset = pd.read_csv(dataset_path)
    items_list = dataset.to_dict(orient="records")

    mongodb_uri = os.getenv("DB_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("DB_NAME", "heart_failure_db")
    database = get_database(uri=mongodb_uri, db_name=db_name)

    records_count = database.count_documents(collection_name="heart_data", query={})

    if records_count > 0:
        print("Los datos ya están en la base de datos.")
        return

    database.insert_many(collection_name="heart_data", documents=items_list)

    print("Datos insertados en la base de datos.")
