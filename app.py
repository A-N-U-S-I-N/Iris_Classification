import streamlit as st
import numpy as np
import joblib
import pandas as pd

model = joblib.load('iris_model.pkl')
scaler = joblib.load('scaler.pkl')

species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

st.title("🌸 Iris Flower Species Prediction")
st.write("Provide the measurements below to predict the iris species.")

sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.0)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]

    predicted_species = species_map[int(prediction)]

    st.success(f"Predicted Species: **{predicted_species}**")

    confidences = {species_map[int(label)]: f"{p*100:.2f}%" for label, p in zip(model.classes_, prob)}

    confidences_df = pd.DataFrame({
        "Species": list(confidences.keys()),
        "Confidence": list(confidences.values())
    })

    confidences_df_chart = pd.DataFrame({
        "Species": list(confidences.keys()),
        "Confidence": [float(v.strip('%')) for v in confidences.values()]
    })
    
    col1, col2 = st.columns(2)

    with col1:
        st.write("Model Confidence Scores (Table):")
        st.table(confidences_df)

    with col2:
        st.write("Model Confidence Scores (Bar Chart):")
        st.bar_chart(confidences_df_chart.set_index("Species"))