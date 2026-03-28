# gru-amazon-review-rating-prediction
GRU-based model for Amazon review rating prediction with NLP preprocessing and SROCC evaluation (PyTorch)
# GRU-based Rating Prediction on Amazon Reviews (NLP)

This project implements a GRU-based deep learning model to predict product review ratings (1–5) from textual data using the Amazon Product Reviews dataset.

---

## 🚀 Features
- Text preprocessing using NLTK:
  - Tokenization
  - Stopword removal
  - Punctuation cleaning
- Vocabulary creation using Torchtext
- Sequence padding for uniform input length
- GRU-based sequence modeling:
  - Custom implementation using `nn.GRUCell`
  - Comparison with `nn.GRU`
- Evaluation using correlation metric (SROCC)

---

## 📂 Dataset
- Amazon Product Reviews dataset (Kaggle)
- Ratings: 1 to 5

---

## 🧠 Model Architecture
- Embedding layer for text representation
- GRU-based sequence model:
  - Variant 1: Manual GRU using `nn.GRUCell`
  - Variant 2: Standard `nn.GRU`
- Fully connected layer for prediction

---

## ⚙️ Training Details
- Optimizer: Adam
- Loss Function: Depends on formulation (classification/regression)
- Train/Val/Test Split: 70/15/15

---

## 📊 Evaluation
- Metric: Spearman Rank Order Correlation Coefficient (SROCC)
- Compared performance of:
  - GRUCell-based model
  - GRU-based model

---

## 🔍 Key Insights
- Sequence modeling improves prediction of review sentiment
- GRU captures contextual dependencies in text
- Correlation-based evaluation provides better ranking insight than accuracy

---

## 🛠️ Tech Stack
- Python
- PyTorch
- Torchtext
- NLTK
- NumPy

---

## ▶️ How to Run

```bash
git clone https://github.com/yourusername/gru-amazon-review-rating-prediction
cd gru-amazon-review-rating-prediction
pip install -r requirements.txt
python train.py
