import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("path/to/saved_tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained("path/to/saved_model")
    return tokenizer, model

tokenizer, model = load_model()

st.title("📮 Spam Message Classifier")
st.write("Type a message below and let BERT decide if it's *spam* or *ham* 🕵️‍♀️📩")

# User input
text = st.text_area("Your message here:")

if st.button("Classify"):
    if text.strip() == "":
        st.warning("Please enter a message, babe 😘")
    else:
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Ensure model runs on the correct device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        label = "💔 Spam" if pred == 1 else "💌 Not Spam"
        st.subheader(f"Prediction: {label}")
