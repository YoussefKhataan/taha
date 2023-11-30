# streamlit_app.py
import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load('text_classification_model.joblib')
vectorizer = joblib.load('text_vectorizer.joblib')

# Streamlit app
def main():
    st.title("Text Classification App")

    # User input
    user_input = st.text_area("Enter your text here:")

    # Prediction button
    if st.button("Predict"):
        if user_input:
            # Vectorize the user input
            user_input_vec = vectorizer.transform([user_input])

            # Make prediction
            prediction = model.predict(user_input_vec)[0]

            # Display result
            st.success(f"Prediction: {prediction}")
        else:
            st.warning("Please enter a text before predicting.")

if __name__ == "__main__":
    main()
