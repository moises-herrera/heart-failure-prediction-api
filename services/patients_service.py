class PatientsService:
    def __init__(self, db_client):
        self.db_client = db_client

    def get_patients(self, filters):
        patients = self.db_client.find_documents(
            collection_name="heart_data", query=filters
        )

        return [self.serialize_patient(patient) for patient in patients]

    def serialize_patient(self, patient_record):
        patient_record["_id"] = str(patient_record["_id"])
        return patient_record


def get_patients_service(db_client):
    return PatientsService(db_client)