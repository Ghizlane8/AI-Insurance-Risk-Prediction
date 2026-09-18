import pandas as pd
import numpy as np
import shap


# ============================================================
# EXPLICATION SHAP DU MODÈLE
# ============================================================

def explain_prediction(model, data, top_n=10):
    """
    Explique une ou plusieurs prédictions du modèle Random Forest
    avec SHAP.

    Parameters
    ----------
    model : sklearn Pipeline
        Pipeline contenant le préprocesseur et le Random Forest.

    data : pandas.DataFrame
        Données brutes correspondant au format utilisé par
        predict_fraud().

    top_n : int
        Nombre de variables importantes à retourner.

    Returns
    -------
    pandas.DataFrame
        Variables et contributions SHAP.
    """

    # --------------------------------------------------------
    # Vérification
    # --------------------------------------------------------

    if data is None or len(data) == 0:
        raise ValueError(
            "Aucun dossier à expliquer."
        )

    data = data.copy()

    # --------------------------------------------------------
    # Récupération du pipeline
    # --------------------------------------------------------

    if not hasattr(model, "named_steps"):
        raise ValueError(
            "Le modèle doit être un Pipeline sklearn."
        )

    if "preprocessor" not in model.named_steps:
        raise ValueError(
            "Le pipeline ne contient pas de preprocessor."
        )

    if "classifier" not in model.named_steps:
        raise ValueError(
            "Le pipeline ne contient pas de classifier."
        )

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # --------------------------------------------------------
    # Préparation des données
    # --------------------------------------------------------

    from src.prediction import prepare_input

    data_prepared = prepare_input(data)

    # --------------------------------------------------------
    # Transformation des variables
    # --------------------------------------------------------

    X_transformed = preprocessor.transform(
        data_prepared
    )

    # --------------------------------------------------------
    # Noms des variables après preprocessing
    # --------------------------------------------------------

    try:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

    except Exception:

        feature_names = [
            f"Feature {i + 1}"
            for i in range(
                X_transformed.shape[1]
            )
        ]

    # --------------------------------------------------------
    # SHAP TreeExplainer
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        classifier
    )

    shap_values = explainer.shap_values(
        X_transformed
    )

    # --------------------------------------------------------
    # Gestion des différentes versions de SHAP
    # --------------------------------------------------------

    if isinstance(shap_values, list):

        # Classification binaire :
        # classe 1 = fraude

        shap_fraud = shap_values[1]

    elif isinstance(shap_values, np.ndarray):

        if shap_values.ndim == 3:

            # Format :
            # observations × features × classes

            shap_fraud = shap_values[:, :, 1]

        else:

            shap_fraud = shap_values

    else:

        shap_fraud = np.asarray(
            shap_values
        )

    # --------------------------------------------------------
    # Explication du premier dossier
    # --------------------------------------------------------

    values = shap_fraud[0]

    explanation = pd.DataFrame({
        "Variable": feature_names,
        "SHAP": values
    })

    # --------------------------------------------------------
    # Valeur absolue = importance
    # --------------------------------------------------------

    explanation["Importance"] = (
        explanation["SHAP"]
        .abs()
    )

    # --------------------------------------------------------
    # Direction de l'influence
    # --------------------------------------------------------

    explanation["Impact"] = np.where(
        explanation["SHAP"] > 0,
        "Augmente le risque",
        "Diminue le risque"
    )

    # --------------------------------------------------------
    # Tri par importance
    # --------------------------------------------------------

    explanation = (
        explanation
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return explanation


# ============================================================
# TOP FACTEURS POSITIFS
# ============================================================

def get_risk_factors(
    model,
    data,
    top_n=5
):
    """
    Retourne les principaux facteurs qui augmentent
    le score de fraude.
    """

    explanation = explain_prediction(
        model,
        data,
        top_n=50
    )

    positive = explanation[
        explanation["SHAP"] > 0
    ].copy()

    positive = (
        positive
        .sort_values(
            "SHAP",
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return positive


# ============================================================
# TOP FACTEURS RÉDUISANT LE RISQUE
# ============================================================

def get_protective_factors(
    model,
    data,
    top_n=5
):
    """
    Retourne les principaux facteurs qui diminuent
    le score de fraude.
    """

    explanation = explain_prediction(
        model,
        data,
        top_n=50
    )

    negative = explanation[
        explanation["SHAP"] < 0
    ].copy()

    negative = (
        negative
        .sort_values(
            "SHAP",
            ascending=True
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return negative