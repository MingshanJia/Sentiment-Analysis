import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import base64

# Load pre-trained model and tokenizer
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

@st.cache_data
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer(MODEL_NAME)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Function to perform sentiment analysis
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    sentiment_label = torch.argmax(outputs.logits, dim=1).item()
    sentiment = {0: "negative", 1: "neutral", 2: "positive"}.get(sentiment_label)
    # Calculate probabilities
    probabilities = torch.softmax(outputs.logits, dim=1).squeeze().tolist()
    return sentiment, probabilities

def get_image_base64_str(file_path):
    with open(file_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded_image}"

st.markdown(
    """
    <style>
    .logo {
        position: absolute;
        bottom: 10px;
        right: 10px;
        z-index: 999;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display the image using the st.markdown function and an HTML img tag
image_base64_str = get_image_base64_str("tafe_nsw_2.jpg")
st.markdown(f'<img src="{image_base64_str}" class="logo" width="100">', unsafe_allow_html=True)

# Streamlit app
st.title("Sentiment Analysis")

# Input text
input_text = st.text_input("Enter text to analyze:", max_chars=256)


# Predict and show output
if input_text:
    sentiment, probabilities = analyze_sentiment(input_text)
    sentiment_color = {"negative": "red", "neutral": "orange", "positive": "green"}.get(sentiment)
    st.markdown(f"<h2>Sentiment: <span style='color: {sentiment_color};'>{sentiment}</span></h2>", unsafe_allow_html=True)
    # Display probabilities
    st.write(f"Probabilities:")
    st.write(f"Negative: {probabilities[0]:.4f}")
    st.write(f"Neutral: {probabilities[1]:.4f}")
    st.write(f"Positive: {probabilities[2]:.4f}")