# 🛡️ Multi-Level Hate Speech Detector

A deep learning framework built to detect and classify hate speech and offensive content across multi-tier text datasets using PyTorch and Hugging Face Transformers.

## 📌 Project Overview
* **Architecture:** Transformer-based fine-tuning (DeBERTa / BERT)
* **Goal:** Multi-level text classification for toxic, offensive, and hate speech detection.
* **Tech Stack:** Python, PyTorch, Hugging Face Transformers, Pandas, Scikit-Learn

## 📁 Repository Structure
```text
├── inference.py       # Run predictions on custom input text
├── prepare_data.py    # Preprocessing and dataset preparation
├── train_tier1.py     # Training loop and model fine-tuning
├── requirements.txt   # Dependencies list
└── .gitignore         # Untracked heavy files & model weights
