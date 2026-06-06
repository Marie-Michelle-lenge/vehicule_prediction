from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import Optional

# ==========================================
# INITIALISATION API
# ==========================================

app = FastAPI(
    title="API DGI - Analyse Prédictive",
    description="API de prédiction du statut fiscal des véhicules",
    version="1.0"
)

# ==========================================
# CHARGEMENT MODELE
# ==========================================

model = joblib.load("model/modele_dgi_pipeline.pkl")

# ==========================================
# BASE DES PREDICTIONS
# ==========================================

predictions_db = []

# ==========================================
# CLASSES
# ==========================================

classes = {
    0: "Non conforme",
    1: "Conforme",
    2: "A risque"
}

# ==========================================
# MODELE DE DONNEES
# ==========================================

class VehicleData(BaseModel):

    type_vehicule: str
    usage: str
    marque: str
    modele: str
    type: str
    genre: str
    couleur: str

    assurance: str
    controle_technique: str
    vignette: str

    puissance: int
    annee_fabrication: int

    valeur_estimee_usd: float
    taxe_circulation_usd: float
    taxe_stationnement_usd: float


# ==========================================
# ENDPOINT ACCUEIL
# ==========================================

@app.get("/")
def home():
    return {
        "message": "API DGI active"
    }


# ==========================================
# ENDPOINT PREDICTION
# ==========================================

@app.post("/predict")
def predict(data: VehicleData):

    input_data = pd.DataFrame([{
        "type_vehicule": data.type_vehicule,
        "usage": data.usage,
        "marque": data.marque,
        "modele": data.modele,
        "type": data.type,
        "genre": data.genre,
        "couleur": data.couleur,

        "assurance": data.assurance,
        "controle_technique": data.controle_technique,
        "vignette": data.vignette,

        "puissance": data.puissance,
        "annee_fabrication": data.annee_fabrication,

        "valeur_estimee_usd": data.valeur_estimee_usd,
        "taxe_circulation_usd": data.taxe_circulation_usd,
        "taxe_stationnement_usd": data.taxe_stationnement_usd
    }])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probabilités
    probabilities = model.predict_proba(input_data)[0]

    result = {
        "type_vehicule": data.type_vehicule,
        "marque": data.marque,
        "modele": data.modele,

        "statut_fiscal": classes[prediction],

        "probabilites": {
            "Non conforme": round(probabilities[0] * 100, 2),
            "Conforme": round(probabilities[1] * 100, 2),
            "A risque": round(probabilities[2] * 100, 2)
        }
    }

    # Sauvegarde mémoire
    predictions_db.append(result)

    return result


# ==========================================
# ENDPOINT LISTE DES PREDICTIONS
# ==========================================

@app.get("/predictions")
def get_predictions(

    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1),

    statut: Optional[str] = None,
    marque: Optional[str] = None,
    type_vehicule: Optional[str] = None

):

    data = predictions_db

    # ======================================
    # FILTRES
    # ======================================

    if statut:
        data = [
            item for item in data
            if item["statut_fiscal"].lower() == statut.lower()
        ]

    if marque:
        data = [
            item for item in data
            if item["marque"].lower() == marque.lower()
        ]

    if type_vehicule:
        data = [
            item for item in data
            if item["type_vehicule"].lower() == type_vehicule.lower()
        ]

    # ======================================
    # PAGINATION
    # ======================================

    start = (page - 1) * limit
    end = start + limit

    paginated_data = data[start:end]

    return {
        "page": page,
        "limit": limit,
        "total": len(data),
        "results": paginated_data
    }