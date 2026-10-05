import streamlit as st
import pandas as pd
import joblib


# =========================================================
# CONFIGURATION DE LA PAGE
# =========================================================

st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="centered"
)


# =========================================================
# CHARGEMENT DU MODÈLE
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("machine_failure_model.pkl")


model = load_model()


# =========================================================
# TITRE
# =========================================================

st.title("⚙️ Machine Failure Prediction")

st.write(
    """
    Cette application utilise un modèle **Random Forest**
    pour estimer le risque de panne d'une machine.

    Entrez les paramètres de fonctionnement ci-dessous.
    """
)

st.divider()


# =========================================================
# INPUTS
# =========================================================

st.subheader("📊 Paramètres de la machine")


col1, col2 = st.columns(2)


# -------------------------
# Colonne gauche
# -------------------------

with col1:

    air_temp = st.number_input(
        "Air temperature [K]",
        min_value=250.0,
        max_value=350.0,
        value=300.0,
        step=0.1
    )

    rpm = st.number_input(
        "Rotational speed [rpm]",
        min_value=500,
        max_value=4000,
        value=1500,
        step=10
    )

    tool_wear = st.number_input(
        "Tool wear [min]",
        min_value=0,
        max_value=300,
        value=100,
        step=1
    )


# -------------------------
# Colonne droite
# -------------------------

with col2:

    process_temp = st.number_input(
        "Process temperature [K]",
        min_value=250.0,
        max_value=400.0,
        value=310.0,
        step=0.1
    )

    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.1
    )

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )


# =========================================================
# ENCODAGE DE TYPE
# =========================================================

type_mapping = {
    "L": 0,
    "M": 1,
    "H": 2
}

machine_type_encoded = type_mapping[machine_type]


# =========================================================
# CONSTRUCTION DES DONNÉES
# =========================================================

machine = pd.DataFrame([{

    "Air temperature [K]": air_temp,

    "Process temperature [K]": process_temp,

    "Rotational speed [rpm]": rpm,

    "Torque [Nm]": torque,

    "Tool wear [min]": tool_wear,

    "Type": machine_type_encoded

}])


st.divider()


# =========================================================
# PRÉDICTION
# =========================================================

if st.button(
    "🔍 Analyser la machine",
    use_container_width=True
):

    # Prédiction 0 / 1
    prediction = model.predict(machine)[0]

    # Probabilité de panne
    probability = model.predict_proba(machine)[0, 1]


    # =====================================================
    # AFFICHAGE PROBABILITÉ
    # =====================================================

    st.subheader("Résultat")

    st.metric(
        label="Probabilité estimée de panne",
        value=f"{probability:.1%}"
    )

    st.progress(float(probability))


    # =====================================================
    # AFFICHAGE PRÉDICTION
    # =====================================================

    if prediction == 1:

        st.error(
            "⚠️ Risque de panne détecté"
        )

        st.write(
            """
            Le modèle classe cette machine dans la catégorie
            **panne potentielle**.
            """
        )

    else:

        st.success(
            "✅ Fonctionnement normal"
        )

        st.write(
            """
            Le modèle classe actuellement cette machine
            dans la catégorie **fonctionnement normal**.
            """
        )


    # =====================================================
    # AFFICHAGE DES DONNÉES
    # =====================================================

    with st.expander("Voir les données envoyées au modèle"):

        st.dataframe(
            machine,
            use_container_width=True
        )


# =========================================================
# INFORMATIONS SUR LE MODÈLE
# =========================================================

st.divider()

with st.expander("ℹ️ Informations sur le modèle"):

    st.write(
        """
        **Modèle :** Random Forest Classifier

        **Variables utilisées :**

        - Air temperature
        - Process temperature
        - Rotational speed
        - Torque
        - Tool wear
        - Machine type

        Le résultat correspond à une estimation statistique
        produite par le modèle.
        """
    )