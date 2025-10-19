import os
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
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
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

    def predict(self, input_data: pd.DataFrame):
        for col, le in self.label_encoders.items():
            if col in input_data.columns:
                input_data[col] = le.transform(input_data[col])

        input_data_scaled = self.scaler.transform(input_data)
        predictions = self.model.predict(input_data_scaled)
        return predictions


def get_classifier():
    dataset_path = "data/heart.csv"
    classifier = HeartDiseaseClassifier(dataset_path)

    classifier.load_and_preprocess_data()
    classifier.train_model()
    classifier.evaluate_model()
    classifier.display_results()
    return classifier
