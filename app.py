from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


COLUMNS = ['Hours_Studied', 'Attendance', 'Parental_Involvement',
        'Access_to_Resources', 'Sleep_Hours', 'Previous_Scores',
        'Motivation_Level', 'Tutoring_Sessions', 'Family_Income',
        'Teacher_Quality', 'Physical_Activity', 'Parental_Education_Level',
        'Distance_from_Home', 'Extracurricular_Activities_Yes',
        'Internet_Access_Yes', 'School_Type_Public', 'Peer_Influence_Neutral',
        'Peer_Influence_Positive', 'Learning_Disabilities_Yes', 'Gender_Male']

NUMERIQUES = {
    "Hours_Studied": (0, 50, 20),
    "Attendance": (0, 100, 70),
    "Sleep_Hours": (2, 10, 6),
    "Previous_Scores": (0, 100, 60),
    "Tutoring_Sessions": (0, 10, 0),
    "Physical_Activity": (0, 10, 3),
}
ORDINALES = {
    "Parental_Involvement": {"Low": 0, "Medium": 1, "High": 2},
    "Access_to_Resources": {"Low": 0, "Medium": 1, "High": 2},
    "Motivation_Level": {"Low": 0, "Medium": 1, "High": 2},
    "Family_Income": {"Low": 0, "Medium": 1, "High": 2},
    "Teacher_Quality": {"Low": 0, "Medium": 1, "High": 2},
    "Parental_Education_Level": {"High School": 0, "College": 1, "Postgraduate": 2},
    "Distance_from_Home": {"Near": 0, "Moderate": 1, "Far": 2},
}
NOMINALES = {
    "Gender": ["Female", "Male"],
    "School_Type": ["Private", "Public"],
    "Extracurricular_Activities": ["No", "Yes"],
    "Internet_Access": ["No", "Yes"],
    "Learning_Disabilities": ["No", "Yes"],
    "Peer_Influence": ["Negative", "Neutral", "Positive"],
}



@st.cache_resource
def charger():
    return joblib.load("/models/final_pipeline.joblib")

st.title("EduPredict : score d'examen")

try:
    modele = charger()
except FileNotFoundError:
    st.error("Modèle n'est pas trouvé")
    st.stop()

ligne = pd.DataFrame(0, index=[0], columns=COLUMNS)

for col, (mini, maxi, defaut) in NUMERIQUES.items():
    ligne[col] = st.slider(col.replace("_"," "), mini, maxi, defaut)
for col, mapping in ORDINALES.items():
    ligne[col] = mapping[st.selectbox(col.replace("_"," "), list(mapping))]
for col, choix in NOMINALES.items():
    nom = f"{col}_{st.selectbox(col.replace("_"," "), choix)}"
    if nom in ligne.columns:
        ligne[nom] = 1

if st.button("Prédire"):
    score = float(modele.predict(ligne[COLUMNS])[0])
    st.metric("Score estimé", f"{score:.1f}")
