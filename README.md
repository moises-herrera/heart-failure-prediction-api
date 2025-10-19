# Heart Failure Prediction API

API REST desarrollada en Flask para predecir la probabilidad de insuficiencia cardíaca en pacientes utilizando un modelo de clasificación de Machine Learning. La API permite obtener datos de pacientes y estadísticas relacionadas con la insuficiencia cardíaca.


## Instalación

1. **Clonar el repositorio**

```bash
git clone https://github.com/moises-herrera/heart-failure-prediction-api.git
cd heart-failure-prediction-api
```

2. **Crear entorno virtual**

```bash
py -m venv venv

venv\Scripts\activate
```

3. **Instalar dependencias**

```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno**

```bash
cp .env.example .env
```

## Uso

### Iniciar el servidor

```bash
py app.py
```

La API estará disponible en: `http://localhost:5000`

### Documentación

- **Swagger UI**: http://localhost:5000/docs