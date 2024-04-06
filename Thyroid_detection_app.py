import streamlit as st
import numpy as np
import pickle
import asyncio

# Install xgboost
import sys
!{sys.executable} -m pip install xgboost

class_mapping = {
    0: 'compensated_hypothyroid',
    1: 'hyperthyroid',
    2: 'negative',
    3: 'primary_hypothyroid'
}

# Rest of your code remains unchanged


def load_model_and_encoder(model_path, encoder_path):
    model = pickle.load(open(model_path, "rb"))
    encoder = pickle.load(open(encoder_path, "rb"))
    return model, encoder

def predict_thyroid_class(model, encoder, input_data):
    predicted_class = model.predict(input_data)
    decoded_class = encoder.inverse_transform(predicted_class)
    mapped_class = class_mapping[decoded_class[0]]
    return mapped_class

async def main():
    st.title("Thyroid Disease Detection")
    st.write("Enter the required information below from the Blood report:")

    age = st.number_input("Enter age:", min_value=0, step=1)
    sex = st.radio("Select sex:", ('Female', 'Male'))
    tsh = st.number_input("Enter TSH value(μIU/ml):")
    t3 = st.number_input("Enter T3 value(ng/dL):")
    t4 = st.number_input("Enter T4 value(μg/dL):")

    sex_numeric = 0 if sex == 'Female' else 1
    input_data = np.array([[age, sex_numeric, tsh, t3, t4]])

    model_path = "tddmodelml.pkl"
    encoder_path = "encoder.pickle"
    model, encoder = load_model_and_encoder(model_path, encoder_path)

    if st.button("Predict"):
        predicted_class = await asyncio.to_thread(predict_thyroid_class, model, encoder, input_data)
        st.write(f"Predicted Class: {predicted_class}")

if __name__ == "__main__":
    asyncio.run(main())
