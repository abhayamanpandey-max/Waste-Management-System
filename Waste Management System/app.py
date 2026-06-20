import os
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

MODEL_FILENAME = 'Waste Management System.h5'
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)


@st.cache_resource(show_spinner=False)
def load_waste_model():
    return load_model(MODEL_PATH)

# Define the class names for prediction output
class_names = ['Recyclable', 'Organic']

# Function to preprocess the image and make a prediction
def predict_waste_type(image_data):
    img = image_data.convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    model = load_waste_model()
    predictions = model.predict(img_array)

    if predictions.ndim == 2 and predictions.shape[1] == 1:
        prob = float(predictions[0][0])
        idx = 1 if prob >= 0.5 else 0
        conf = prob if idx == 1 else 1.0 - prob
    else:
        idx = int(np.argmax(predictions, axis=1)[0])
        conf = float(predictions[0][idx])

    return class_names[idx], conf

# Streamlit app layout
st.title("Waste Classification App")
st.write("Upload an image to classify it as Organic or Recyclable waste.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Make prediction on button click
    if st.button('Classify'):
        with st.spinner('Classifying...'):
            try:
                label, conf = predict_waste_type(image)
                st.success(f"Prediction: This waste is **{label}** (confidence {conf:.2f})")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.markdown("--- App Developed for Waste Classification --- ")

# Validate model presence early for clearer errors
if not os.path.isfile(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}")

# To run this Streamlit app:
# 1. Save this code into a Python file (e.g., `app.py`).
# 2. Open your terminal or command prompt.
# 3. Navigate to the directory where you saved `app.py`.
# 4. Run the command: `streamlit run app.py`
# 5. The app will open in your web browser.