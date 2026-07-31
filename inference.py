import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = os.path.abspath(os.path.join("models", "deberta_hate_speech"))

def load_model():
    print(f"Loading fine-tuned DeBERTa model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
    model.eval()
    return tokenizer, model

def predict(text, tokenizer, model):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=128
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_class].item() * 100

    label_map = {0: "Non-Hate Speech / Safe", 1: "Hate Speech / Toxic"}
    return label_map[pred_class], confidence

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Model path not found: {MODEL_PATH}")
        print("Please run training to completion first.\n")
    else:
        tokenizer, model = load_model()
        
        print("\n--- Hate Speech Detector Ready ---")
        print("Type a sentence to test (or type 'exit' to quit):\n")
        
        while True:
            user_input = input("Enter text: ")
            if user_input.lower() == "exit":
                break
            if user_input.strip() == "":
                continue
                
            label, conf = predict(user_input, tokenizer, model)
            print(f"-> Prediction: {label} ({conf:.2f}% confidence)\n")