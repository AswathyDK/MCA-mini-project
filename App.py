import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
# Load the fine-tuned model
model = load_model('fake_logodetection_vgg16_finetuned_best_improved.h5')
# Preprocess image to match the model’s input size and scaling
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to 224x224
    img_array = img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array
# Prediction function
def predict_image(img):
    img_array = preprocess_image(img)  # Preprocess the image
    prediction = model.predict(img_array)  # Get prediction
    st.write(f"Raw prediction output: {prediction}")  # Show raw prediction
    confidence = prediction[0][0]  # Extract confidence
    threshold = 0.5  # Consistent threshold
    predicted_class = 'Genuine' if confidence > threshold else 'Fake'  # Determine class
    #st.write(f"Prediction confidence: {confidence:.4f}")  # Show confidence level
    return predicted_class
# Background image URL
background_image_url = "imageUI.png"  # Example URL
# Streamlit UI with background image
st.markdown(
    f"""
    <style>
    body {{
        background-image: url('{background_image_url}');  /* Use your image URL or path */
        background-size: cover;
        background-repeat: no-repeat;
        color: white;  /* Change text color to white for better visibility */
        font-family: 'Arial', sans-serif;  /* Change font */
    }}
    .stButton {{
        background-color: #4CAF50;  /* Green background for the button */
        color: white;  /* Text color for the button */
        border: none;  /* No border */
        padding: 10px 20px;  /* Padding for the button */
        text-align: center;  /* Center align text */
        text-decoration: none;  /* No underline */
        display: inline-block;  /* Inline block display */
        font-size: 16px;  /* Font size */
        margin: 4px 2px;  /* Margins */
        cursor: pointer;  /* Pointer cursor on hover */
        border-radius: 5px;  /* Rounded corners */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Fake Logo Detection")
st.write("Upload an image to determine if it is genuine or fake.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_img(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Predict"):
        result = predict_image(image)
        st.write(f"The image is predicted as: **{result}**")
