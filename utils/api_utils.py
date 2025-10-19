from flask import request


def parse_filters(query_params):
    query_params = request.args

    validate_filters(query_params)

    age = query_params.get("age")
    gender = query_params.get("gender")
    heart_disease = query_params.get("heart_disease")

    filters = {}

    if age:
        filters["Age"] = int(age)
    if gender:
        filters["Sex"] = gender
    if heart_disease:
        filters["HeartDisease"] = int(heart_disease)

    return filters


def validate_filters(filters):
    valid_keys = {"age", "gender", "heart_disease"}
    for key in filters.keys():
        if key not in valid_keys:
            raise ValueError(
                {
                    "error": f"Filtro inválido: {key}. Los filtros válidos son: {', '.join(valid_keys)}"
                },
                400,
            )


def validate_patient_data(data):
    required_fields = {
        "Age",
        "Sex",
        "ChestPainType",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "RestingECG",
        "MaxHR",
        "ExerciseAngina",
        "Oldpeak",
        "ST_Slope",
    }
    missing_fields = []

    for field in required_fields:
        if field not in data:
            missing_fields.append(field)

    if missing_fields:
        raise ValueError(
            {"error": f"Faltan campos obligatorios: {', '.join(missing_fields)}"}, 400
        )
