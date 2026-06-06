import streamlit as st
import pandas as pd
import joblib

# Charger le pipeline complet
model = joblib.load("C:/Users/LE BON MEC/Documents/cours MM/TFC/model/modele_dgi_pipeline.pkl")

st.title(" Système Intelligent DGI - Analyse Fiscale")

st.header(" Informations du véhicule")

# ========================
# INPUTS
# ========================

type_vehicule = st.selectbox("Type de véhicule", ["Voiture", "Bus", "Camion"])
usage = st.selectbox("Usage", ["Privé", "Transport public"])

marque = st.text_input("Marque", "Toyota")
modele = st.text_input("Modèle", "Corolla")

genre = st.selectbox("Genre", ["SUV", "Berline", "JEEP"])
couleur = st.text_input("Couleur", "Noir")

assurance = st.selectbox("Assurance", ["Valide", "Expiré"])
controle = st.selectbox("Contrôle technique", ["Valide", "Expiré"])
vignette = st.selectbox("Vignette", ["Valide", "Expiré"])

puissance = st.number_input("Puissance", min_value=0, value=100)
annee = st.number_input("Année de fabrication", min_value=1900, value=2015)

valeur = st.number_input("Valeur estimée USD", min_value=0.0, value=5000.0)
taxe_circ = st.number_input("Taxe circulation USD", min_value=0.0, value=100.0)
taxe_stat = st.number_input("Taxe stationnement USD", min_value=0.0, value=50.0)

# ========================
# PREDICTION
# ========================

if st.button("🔍 Analyser le véhicule"):

    input_data = pd.DataFrame([{
        "type": type_vehicule,
        "usage": usage,
        "marque": marque,
        "modele": modele,
        "genre": genre,
        "couleur": couleur,
        "assurance": assurance,
        "controle_technique": controle,
        "vignette": vignette,
        "puissance": puissance,
        "annee_fabrication": annee,
        "valeur_estimee_usd": valeur,
        "taxe_circulation_usd": taxe_circ,
        "taxe_stationnement_usd": taxe_stat
    }])

    # ========================
    # CORRECTION TYPES
    # ========================
    cat_cols = ["type", "usage", "marque", "modele", "genre", "couleur",
                "assurance", "controle_technique", "vignette"]

    for col in cat_cols:
        input_data[col] = input_data[col].astype(str)

    num_cols = ["puissance", "annee_fabrication", "valeur_estimee_usd",
                "taxe_circulation_usd", "taxe_stationnement_usd"]

    for col in num_cols:
        input_data[col] = pd.to_numeric(input_data[col], errors="coerce")

    # ========================
    # PREDICTION
    # ========================
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    # ========================
    # CLASSES
    # ========================
    classes = [" Non conforme", " Conforme", " À risque"]

    # ========================
    # AFFICHAGE RESULTAT
    # ========================
    st.subheader("📊 Résultat de l'analyse")

    st.success(f"Statut : {classes[prediction]}")

    # ========================
    # PROBABILITÉS
    # ========================
    st.write("### 🔢 Probabilités :")

    st.write(f"Non conforme : {round(proba[0]*100, 2)} %")
    st.write(f"Conforme : {round(proba[1]*100, 2)} %")
    st.write(f"À risque : {round(proba[2]*100, 2)} %")

    # ========================
    # SCORE DE RISQUE
    # ========================
    risk_score = proba[0]*100 + proba[2]*50

    st.write(f"###  Score de risque : {round(risk_score,2)}/100")

    # ========================
    # INTERPRÉTATION DGI
    # ========================
    st.write("###  Interprétation")

    if prediction == 0:
        st.error(" Véhicule NON CONFORME → Intervention immédiate recommandée")
    elif prediction == 2:
        st.warning(" Véhicule À RISQUE → Surveillance recommandée")
    else:
        st.success(" Véhicule CONFORME → Aucun problème détecté")