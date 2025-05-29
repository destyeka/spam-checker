import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gdown

# Shareable link (make sure it's public or "anyone with the link")
gdown.download("https://drive.google.com/file/d/131e-hQWCBBYvqmuuVSGF_nwxCw9grcSg/view?usp=drive_link", output="model.zip", quiet=False)


st.title("📮 Spam Message Classifier")
st.write("Type a message below and let BERT decide if it's *spam* or *ham* 🕵️‍♀️📩")

# User input
text = st.text_area("Your message here:")

if st.button("Classify"):
    if text.strip() == "":
        st.warning("Please enter a message, babe 😘")
    else:

        label = "💔 Spam" if pred == 1 else "💌 Not Spam"
        st.subheader(f"Prediction: {label}")
