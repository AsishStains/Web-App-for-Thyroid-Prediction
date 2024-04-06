import numpy as np
import pickle
import streamlit as st
import sklearn

# Load the Random Forest model and the encoder
@st.cache_data()
def load_model():
    rf_model = pickle.load(open("tddmodelml_rf.pkl", "rb"))
    encoder_rf = pickle.load(open("encoder_rf.pickle", "rb"))
    return rf_model, encoder_rf



# Function to predict thyroid class based on user input
def predict_thyroid_class_rf(model, encoder):
    # Input values from the user
    age = st.number_input("Enter age:", min_value=0, step=1)
    sex = st.radio("Select sex:", ('Female', 'Male'))
    tsh = st.number_input("Enter TSH value(μIU/ml):")
    t3 = st.number_input("Enter T3 value(ng/dL):")
    t4 = st.number_input("Enter T4 value(μg/dL):")

    # Check if any of the inputs are empty
    if not age or not tsh or not t3 or not t4:
        st.warning("Please fill in all the fields.")
        return

    try:
        # Convert input values to appropriate data types
        age = float(age)
        tsh = float(tsh)
        t3 = float(t3)
        t4 = float(t4)
    except ValueError:
        st.warning("Invalid input. Please enter valid numeric values.")
        return

    # Convert sex to numerical value (0 for 'F', 1 for 'M')
    sex_numeric = 0 if sex.upper() == 'F' else 1

    # Create a numpy array with the user input
    input_data = np.array([[age, sex_numeric, tsh, t3, t4]])

    # Make predictions using the loaded model
    predicted_class = model.predict(input_data)

    # Convert predicted_class to integer type
    predicted_class = predicted_class.astype(int)

    # Decode the predicted class using the label encoder
    decoded_class = encoder.inverse_transform(predicted_class)

    # Map the decoded class to the corresponding label
    mapped_class = class_mapping[decoded_class[0]]

    # Print the result
    st.write("Predicted Class:", mapped_class)

if __name__ == "__main__":
    # Load the trained model and encoder
    rf_model, encoder_rf = load_model()

    # Define class mapping
    class_mapping = {0: 'compensated_hypothyroid', 1: 'hyperthyroid', 2: 'negative', 3: 'primary_hypothyroid'}

    # Title of the web app
    st.title('Thyroid Disease Detection')

    # Function call to predict thyroid class
    predict_thyroid_class_rf(rf_model, encoder_rf)
