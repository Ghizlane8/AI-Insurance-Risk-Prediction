import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# Chemins du projet
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "fraud_detection_model.pkl"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "fraud_threshold.pkl"


# ============================================================
# Chargement du modèle
# ============================================================

def load_model():
    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    return model, threshold


# ============================================================
# Préparation des données
# ============================================================

def prepare_input(df):

    df = df.copy()

    # Remplacer les valeurs ?
    df = df.replace("?", np.nan)

    # Valeurs manquantes
    if "authorities_contacted" in df.columns:
        df["authorities_contacted"] = (
            df["authorities_contacted"]
            .fillna("Unknown")
        )

    # Dates
    df["policy_bind_date"] = pd.to_datetime(
        df["policy_bind_date"]
    )

    df["incident_date"] = pd.to_datetime(
        df["incident_date"]
    )

    # Variables temporelles
    df["policy_bind_year"] = (
        df["policy_bind_date"].dt.year
    )

    df["policy_bind_month"] = (
        df["policy_bind_date"].dt.month
    )

    df["incident_year"] = (
        df["incident_date"].dt.year
    )

    df["incident_month"] = (
        df["incident_date"].dt.month
    )

    # Durée de la police
    df["policy_duration_days"] = (
        df["incident_date"]
        - df["policy_bind_date"]
    ).dt.days

    # Âge du véhicule
    df["vehicle_age"] = (
        df["incident_date"].dt.year
        - df["auto_year"]
    )

    # ========================================================
    # Période de la journée
    # ========================================================

    def get_time_period(hour):

        if 6 <= hour < 12:
            return "Morning"

        elif 12 <= hour < 18:
            return "Afternoon"

        elif 18 <= hour < 24:
            return "Evening"

        else:
            return "Night"

    df["incident_period"] = (
        df["incident_hour_of_the_day"]
        .apply(get_time_period)
    )

    # ========================================================
    # Colonnes supprimées lors de l'entraînement
    # ========================================================

    columns_to_drop = [
        "policy_number",
        "incident_location",
        "insured_zip",
        "policy_bind_date",
        "incident_date",
        "_c39",

        # Variables de montant
        "total_claim_amount",
        "injury_claim",
        "property_claim",
        "vehicle_claim",

        # Target
        "fraud_reported"
    ]

    df = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    return df


# ============================================================
# Prédiction de fraude
# ============================================================

def predict_fraud(data):

    # Charger le modèle et le seuil
    model, threshold = load_model()

    # Préparer les données
    data_prepared = prepare_input(data)

    # Calculer la probabilité de fraude
    probability = model.predict_proba(
        data_prepared
    )[:, 1]

    # Appliquer le seuil optimisé
    prediction = (
        probability >= threshold
    ).astype(int)

    # Créer le résultat
    result = pd.DataFrame({
        "fraud_probability": probability,
        "prediction": prediction
    })

    # Niveau de risque
    result["risk_level"] = (
        result["fraud_probability"]
        .apply(
            lambda x:
                "Faible"
                if x < 0.20
                else
                "Modéré"
                if x < 0.50
                else
                "Élevé"
        )
    )

    # Décision finale
    result["decision"] = (
        result["prediction"]
        .map({
            0: "Non fraude",
            1: "Fraude"
        })
    )

    return result