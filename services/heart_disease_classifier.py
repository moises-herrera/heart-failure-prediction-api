import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns


class HeartDiseaseClassifier:
    def __init__(
        self,
        dataset_path: str,
        model_path: str = "ai_models/heart_disease_model.pkl",
    ):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = {}
        self.results = {}

    def load_and_preprocess_data(self):
        self.df = pd.read_csv(self.dataset_path)
        self.df = self.df.sort_index(axis=1)
        print(
            f"\nDataset cargado: {self.df.shape[0]} filas, {self.df.shape[1]} columnas"
        )

        categorical_columns = [
            "Sex",
            "ChestPainType",
            "RestingECG",
            "ExerciseAngina",
            "ST_Slope",
        ]

        for col in categorical_columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le

        X = self.df.drop("HeartDisease", axis=1)
        y = self.df["HeartDisease"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print(
            f"Datos divididos: {len(self.X_train)} entrenamiento, {len(self.X_test)} prueba"
        )

    def train_model(self):
        self.model = GaussianNB()
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        self.results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "y_pred": y_pred,
        }

    def display_results(self):
        print("\nClassification Report:")
        print(
            classification_report(
                self.y_test,
                self.results["y_pred"],
                target_names=["No Disease", "Heart Disease"],
            )
        )

        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, self.results["y_pred"])
        print(cm)

        self.results["confusion_matrix"] = cm

        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Disease", "Heart Disease"],
            yticklabels=["No Disease", "Heart Disease"],
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        if not os.path.exists("results"):
            os.makedirs("results")

        plt.savefig("results/confusion_matrix.png")

    def save_model(self):
        model_dir = os.path.dirname(self.model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "results": self.results,
        }

        with open(self.model_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"\nModelo guardado exitosamente en: {self.model_path}")

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No se encontró el modelo en: {self.model_path}")

        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.label_encoders = model_data["label_encoders"]
        self.results = model_data.get("results", {})

        print(f"\nModelo cargado exitosamente desde: {self.model_path}")

    def predict(self, input_data: pd.DataFrame):
        for col, le in self.label_encoders.items():
            if col in input_data.columns:
                input_data[col] = le.transform(input_data[col])

        input_data_scaled = self.scaler.transform(input_data)
        predictions = self.model.predict(input_data_scaled)
        return predictions
    
    def get_metrics(self):
        cm = self.results.get("confusion_matrix")

        return {
            "accuracy": self.results.get("accuracy"),
            "precision": self.results.get("precision"),
            "recall": self.results.get("recall"),
            "f1_score": self.results.get("f1_score"),
            "true_negatives": float(cm[0][0]),
            "false_positives": float(cm[0][1]),
            "false_negatives": float(cm[1][0]),
            "true_positives": float(cm[1][1]),
        }


def get_classifier(force_retrain=False) -> HeartDiseaseClassifier:
    dataset_path = "data/heart.csv"
    model_path = "ai_models/heart_disease_model.pkl"

    classifier = HeartDiseaseClassifier(dataset_path, model_path)

    if os.path.exists(model_path) and not force_retrain:
        print("\nCargando modelo existente...")
        classifier.load_model()
    else:
        if force_retrain:
            print("\nReentrenando modelo...")
        else:
            print("\nNo se encontró modelo guardado. Entrenando nuevo modelo...")

        classifier.load_and_preprocess_data()
        classifier.train_model()
        classifier.evaluate_model()
        classifier.display_results()
        classifier.save_model()

    return classifier
