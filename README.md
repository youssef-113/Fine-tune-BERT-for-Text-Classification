# 🧠 Fine-tuned BERT for Sentiment Classification

This project fine-tunes a pre-trained BERT model using Hugging Face Transformers to perform sentiment analysis on movie reviews (binary classification: Positive or Negative). It also includes a polished **Streamlit app** to interactively test the model in real time.

![Streamlit Banner](https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.svg)

---

## 🚀 Live Demo

🔗 [Click here to try the deployed app](https://streamlit.io/) *(replace with your actual link after deployment)*

---

## 🧾 Project Summary

* **Model:** BERT (bert-base-uncased)
* **Dataset:** IMDB reviews (binary sentiment)
* **Frameworks:** Hugging Face Transformers, PyTorch
* **Interface:** Streamlit

---

## 📁 Folder Structure

```
Fine-tune-BERT-for-Text-Classification/
├── bert_model/                   # Model weights and tokenizer files
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── special_tokens_map.json
├── bert_sentiment_app.py        # Streamlit app
├── requirements.txt             # App dependencies
└── README.md                    # This file
```

---

## 📦 Setup Locally

```bash
# Clone the repository
git clone https://github.com/youssef-113/Fine-tune-BERT-for-Text-Classification.git
cd Fine-tune-BERT-for-Text-Classification

# Create environment and install requirements
pip install -r requirements.txt

# Run the app
streamlit run bert_sentiment_app.py
```

---

## 📊 Sample Results

| Review                                         | Sentiment | Confidence |
| ---------------------------------------------- | --------- | ---------- |
| "This movie was absolutely amazing!"           | Positive  | 98.2%      |
| "I wasted two hours of my life watching this." | Negative  | 95.4%      |

---

## 🧠 Insight & Conclusion

* **BERT performs well** with minimal tuning on binary text classification.
* Confidence scores can guide interpretation and highlight uncertainty.
* Using Hugging Face `Trainer` simplifies fine-tuning significantly.
* The Streamlit app makes model usage accessible for non-technical users.

### 🔍 Key Takeaway:

> Fine-tuning a powerful pre-trained model like BERT yields strong results with relatively little data and infrastructure. Deploying the model as an interactive web app bridges the gap between research and real-world usage.

---

## ✨ Tools Used

* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [PyTorch](https://pytorch.org/)
* [Streamlit](https://streamlit.io/)

---

## 👨‍💻 Author

**Youssef Bassiony**
[LinkedIn](https://www.linkedin.com/in/youssef-bassiony) | [GitHub](https://github.com/youssef-113)

Feel free to ⭐ the repo or share feedback!

