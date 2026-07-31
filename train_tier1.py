import os
import torch
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)

MODEL_NAME = "microsoft/deberta-v3-base"
OUTPUT_MODEL_DIR = os.path.join("models", "deberta_hate_speech")

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    
    # Fast Demo Subset: Taking 500 train samples and 100 validation samples
    print("Loading fast demo subset of datasets...")
    train_df = pd.read_csv(os.path.join("data", "train_clean.csv")).sample(500, random_state=42)
    val_df = pd.read_csv(os.path.join("data", "val_clean.csv")).sample(100, random_state=42)
    
    print("Loading DeBERTa-v3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("Tokenizing datasets...")
    train_encodings = tokenizer(train_df['comment_text'].tolist(), truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_df['comment_text'].tolist(), truncation=True, padding=True, max_length=128)
    
    train_dataset = TextDataset(train_encodings, train_df['label'].tolist())
    val_dataset = TextDataset(val_encodings, val_df['label'].tolist())
    
    print("Loading DeBERTa-v3 base model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,              # Single epoch for ultra-fast training
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        max_grad_norm=1.0,               # Prevents NaN loss explosion
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print("--- Starting Tier 1 Fast Demo Fine-Tuning ---")
    trainer.train()
    
    print(f"Saving fine-tuned model to '{OUTPUT_MODEL_DIR}'...")
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    print("Training complete and model saved successfully!")

if __name__ == "__main__":
    train()