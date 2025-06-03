import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("destyeka/indonesian_spam_checker")
    model = AutoModelForSequenceClassification.from_pretrained("destyeka/indonesian_spam_checker")
    return tokenizer, model

tokenizer, model = load_model()

st.title("📮 Spam Checker!")
st.write("Masukin pesan lalu lihat hasilnya spam apa bukan!")

# User input
text = st.text_area("Pesannya taro sini...")

if st.button("Cek"):
    if text.strip() == "":
        st.warning("Pesannya belum dimasukin!")
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

        if prediction == "1":
            st.error("🚨 This message is classified as **SPAM**.")
        else:
            st.success("✅ This message is classified as **NOT SPAM**.")

