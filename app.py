import os
from flask import Flask, request, jsonify, Blueprint
from flasgger import Swagger
from utils.process_dataset import process_dataset
from database.database import get_database
from services.patients_service import get_patients_service
from services.statistics_service import get_statistics_service
from services.heart_disease_classifier import get_classifier
import pandas as pd
from dotenv import load_dotenv
from utils.api_utils import parse_filters, validate_patient_data

load_dotenv()

app = Flask(__name__)

db = get_database(
    uri=os.getenv("DB_URI", "mongodb://localhost:27017/"),
    db_name=os.getenv("DB_NAME", "heart_failure_db"),
)

heart_disease_classifier = None

api_bp = Blueprint("api", __name__, url_prefix="/api")

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs",
}

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Heart Failure Prediction API",
        "description": "API para predecir enfermedades cardíacas y consultar estadísticas de pacientes",
        "version": "1.0.0",
    },
    "basePath": "/",
    "schemes": ["http", "https"],
    "tags": [
        {"name": "Health", "description": "Endpoints de salud de la API"},
        {"name": "Patients", "description": "Gestión de información de pacientes"},
        {"name": "Statistics", "description": "Estadísticas y análisis de datos"},
        {
            "name": "Prediction",
            "description": "Predicción de enfermedades cardíacas con Machine Learning",
        },
    ],
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)


@app.get("/")
def root():
    return jsonify(
        {"message": "Heart Failure Prediction API", "docs": "/docs", "api_base": "/api"}
    )


@api_bp.get("/health")
def health_check():
    """
    Verificar el estado de salud de la API
    ---
    tags:
      - Health
    responses:
      200:
        description: Estado de la API
        schema:
          type: object
          properties:
            status:
              type: string
              example: "ok"
    """
    return jsonify({"status": "ok"})


@api_bp.get("/patients")
def get_patients():
    """
    Obtener lista de pacientes con filtros opcionales
    ---
    tags:
      - Patients
    parameters:
      - name: age
        in: query
        type: integer
        required: false
        description: Filtrar por edad específica
        example: 50
      - name: gender
        in: query
        type: string
        required: false
        description: Filtrar por género (M/F)
        example: "M"
      - name: heart_disease
        in: query
        type: integer
        required: false
        description: Filtrar por presencia de enfermedad cardíaca (0/1)
        example: 1
    responses:
      200:
        description: Lista de pacientes
        schema:
          type: array
          items:
            type: object
            properties:
              Age:
                type: integer
              Sex:
                type: string
              ChestPainType:
                type: string
              RestingBP:
                type: integer
              Cholesterol:
                type: integer
              FastingBS:
                type: integer
              RestingECG:
                type: string
              MaxHR:
                type: integer
              ExerciseAngina:
                type: string
              Oldpeak:
                type: number
              ST_Slope:
                type: string
              HeartDisease:
                type: integer
      400:
        description: Parametros invalidos
        schema:
          type: object
          properties:
            error:
                type: string
      500:
        description: Error en el servidor
        schema:
          type: object
          properties:
            error:
              type: string
    """
    try:
        filters = parse_filters(request.args)
        patients = get_patients_service(db).get_patients(filters)
        return jsonify(patients)
    except ValueError as ve:
        (error, status_code) = ve.args
        return jsonify(error), status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.get("/stats")
def get_stats():
    """
    Obtener estadísticas de pacientes
    ---
    tags:
      - Statistics
    parameters:
      - name: age
        in: query
        type: integer
        required: false
        description: Filtrar estadísticas por edad específica
        example: 50
      - name: gender
        in: query
        type: string
        required: false
        description: Filtrar estadísticas por género (M/F)
        example: "M"
      - name: heart_disease
        in: query
        type: integer
        required: false
        description: Filtrar estadísticas por presencia de enfermedad cardíaca (0/1)
        example: 1
    responses:
      200:
        description: Estadísticas de pacientes
        schema:
          type: object
          properties:
            total_patients:
              type: integer
              description: Número total de pacientes
            avg_age:
              type: number
              description: Edad promedio
            gender_distribution:
              type: object
              description: Distribución por género
            heart_disease_rate:
              type: number
              description: Tasa de enfermedad cardíaca
      400:
        description: Parametros invalidos
        schema:
          type: object
          properties:
            error:
                type: string
      500:
        description: Error en el servidor
        schema:
          type: object
          properties:
            error:
              type: string
    """
    try:
        filters = parse_filters(request.args)
        stats = get_statistics_service(db).get_patient_statistics(filters)
        return jsonify(stats)
    except ValueError as ve:
        (error, status_code) = ve.args
        return jsonify(error), status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.post("/predict")
def predict():
    """
    Predecir enfermedad cardíaca usando Machine Learning
    ---
    tags:
      - Prediction
    parameters:
      - name: body
        in: body
        required: true
        description: Datos del paciente para realizar la predicción
        schema:
          type: object
          required:
            - Age
            - Sex
            - ChestPainType
            - RestingBP
            - Cholesterol
            - FastingBS
            - RestingECG
            - MaxHR
            - ExerciseAngina
            - Oldpeak
            - ST_Slope
          properties:
            Age:
              type: integer
              description: Edad del paciente
              example: 55
            Sex:
              type: string
              description: Género del paciente (M/F)
              example: "M"
            ChestPainType:
              type: string
              description: Tipo de dolor de pecho (ATA/NAP/ASY/TA)
              example: "ASY"
            RestingBP:
              type: integer
              description: Presión arterial en reposo (mm Hg)
              example: 140
            Cholesterol:
              type: integer
              description: Colesterol sérico (mm/dl)
              example: 250
            FastingBS:
              type: integer
              description: Glucosa en ayunas > 120 mg/dl (1/0)
              example: 1
            RestingECG:
              type: string
              description: Resultados del electrocardiograma en reposo (Normal/ST/LVH)
              example: "Normal"
            MaxHR:
              type: integer
              description: Frecuencia cardíaca máxima alcanzada
              example: 150
            ExerciseAngina:
              type: string
              description: Angina inducida por ejercicio (Y/N)
              example: "Y"
            Oldpeak:
              type: number
              description: Depresión del ST inducida por ejercicio
              example: 2.5
            ST_Slope:
              type: string
              description: Pendiente del segmento ST (Up/Flat/Down)
              example: "Flat"
    responses:
      200:
        description: Predicción exitosa
        schema:
          type: object
          properties:
            heart_disease_prediction:
              type: integer
              description: Predicción de enfermedad cardíaca (0=No, 1=Sí)
              example: 0
      400:
        description: Datos invalidos
        schema:
          type: object
          properties:
            error:
                type: string
      500:
        description: Error en el servidor
        schema:
          type: object
          properties:
            error:
              type: string
    """
    try:
        data = request.json
        validate_patient_data(data)

        input_data = pd.DataFrame([data])
        input_data = input_data.sort_index(axis=1)
        predictions = heart_disease_classifier.predict(input_data)
        prediction = {"heart_disease_prediction": int(predictions[0])}
        return jsonify(prediction)
    except ValueError as ve:
        (error, status_code) = ve.args
        return jsonify(error), status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


app.register_blueprint(api_bp)


if __name__ == "__main__":
    process_dataset()
    heart_disease_classifier = get_classifier()

    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
