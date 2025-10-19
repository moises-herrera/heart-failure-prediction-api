import pandas as pd

class StatisticsService:
    def __init__(self, db_client):
        self.db_client = db_client

    def get_patient_statistics(self, filters):
        patients = self.db_client.find_documents(
            collection_name="heart_data", query=filters
        )
        df = pd.DataFrame(patients)

        if df.empty:
            return {"message": "No hay datos que coincidan con los filtros proporcionados."}

        stats = {
            "total_patients": len(df),
            "avg_age": df["Age"].mean(),
            "avg_resting_bp": df["RestingBP"].mean(),
            "avg_cholesterol": df["Cholesterol"].mean(),
            "avg_max_heart_rate": df["MaxHR"].mean(),
        }

        return stats
    

def get_statistics_service(db_client):
    return StatisticsService(db_client)