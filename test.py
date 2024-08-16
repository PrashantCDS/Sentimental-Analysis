import streamlit as st
import os

# Check if gdown is installed, and install it if not
try:
    import gdown
except ModuleNotFoundError:
    os.system('pip install gdown')
    import gdown

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Function to download the model from Google Drive
@st.cache_resource
def download_model_from_drive():
    # Google Drive shared link (ensure the link is set to "Anyone with the link")
    file_id = '1960SbLOZsvujYULTcTmvWtfSOgFO19mX'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    # Download the file
    output = 'lstm.h5'
    gdown.download(url, output, quiet=False)
    
    try:
        # Attempt to load the model
        model = load_model(output)
        return model
    except OSError as e:
        st.error(f"Failed to load model: {e}")
        return None

# Load the tokenizer from a pickle file
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Download and load the model
model = download_model_from_drive()

if model is not None:
    # Define max_length based on your training configuration
    max_length = 32

    def predict_sentiment(input_text, tokenizer, model, max_length):
        # Preprocess the input text
        input_sequence = tokenizer.texts_to_sequences([input_text])
        padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length, padding='post')

        # Get the prediction
        prediction = model.predict(padded_input_sequence)

        # Convert the prediction to sentiment label
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        predicted_label_index = np.argmax(prediction)
        predicted_sentiment = sentiment_labels[predicted_label_index]

        return predicted_sentiment

    # Define emojis for each sentiment
    emojis = {
        'Negative': '😡',
        'Neutral': '😐',
        'Positive': '😄'
    }

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .main {
            background-image: url('https://www.shutterstock.com/image-illustration/modern-guard-shield-3d-render-260nw-2010381701.jpg');
            background-size: cover;
            background-position: center;
            color: white;
        }
        .reportview-container {
            background: none;
        }
        h1 {
            color: #ffffff;
            font-family: 'Roboto', sans-serif;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }
        .stTextArea {
            border: 1px solid #ffffff;
            border-radius: 8px;
            padding: 10px;
            color: black;
        }
        .stButton>button {
            color: white !important;
            background-color: #f76c6c !important;
            border-radius: 8px;
            border: none;
            padding: 0.5em 1em;
            cursor: pointer;
        }
        .stButton>button:hover {
            color: #f76c6c !important;
            background-color: white !important;
            border: 1px solid #f76c6c !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar for additional info
    st.sidebar.title("About")
    st.sidebar.info("""
    This is a sentiment analysis app. 
    You can enter any text and get a prediction of whether the sentiment is Negative, Neutral, or Positive.
    """)

    # Main app content
    st.title('Sentiment Analysis App')
    st.write("Enter some text and get the predicted sentiment:")

    # Text input from the user
    input_text = st.text_area("Input Text", placeholder="Type your message here...")

    if st.button("Predict Sentiment"):
        if input_text:
            predicted_sentiment = predict_sentiment(input_text, tokenizer, model, max_length)
            sentiment_emoji = emojis[predicted_sentiment]
            st.write(f"Predicted Sentiment: {sentiment_emoji} **{predicted_sentiment}**", unsafe_allow_html=True)
            st.markdown(f"<h1 style='font-size: 120px;'>{sentiment_emoji}</h1>", unsafe_allow_html=True)
        else:
            st.write("Please enter some text.")

    # Footer
    st.markdown(
        """
        <hr>
        <footer>
            <p style='text-align: center;'>Made with ❤️ using Streamlit</p>
        </footer>
        """,
        unsafe_allow_html=True
    )
else:
    st.error("The model could not be loaded. Please try again later.")
