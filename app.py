import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Page config
st.set_page_config(page_title="BERT Sentiment Classifier", layout="centered")
st.title("🗣️ BERT Sentiment Analysis App")
st.markdown("Analyze the sentiment of your text using a fine-tuned BERT model.")

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("./bert_model")
    tokenizer = AutoTokenizer.from_pretrained("./bert_model")
    return tokenizer, model

tokenizer, model = load_model()

# User input
text = st.text_area("Enter a sentence for sentiment analysis", height=150)

if st.button("Analyze Sentiment"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            sentiment = torch.argmax(probs, dim=1).item()
            labels = ["Negative", "Positive"]

            st.subheader("Result")
            st.write(f"**Sentiment:** {labels[sentiment]}")
            st.write("**Confidence:** {:.2f}%".format(probs[0][sentiment].item() * 100))

            st.progress(probs[0][sentiment].item())

st.markdown("""
---
Model powered by Hugging Face Transformers and Streamlit.  
Ensure your model files (config.json, model.safetensors, tokenizer) are in a directory named `bert_model/`.
""")
