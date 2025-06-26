import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load fine-tuned model and tokenizer
MODEL_PATH = "./finetuned_model"  # or replace with HuggingFace repo if pushed online
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

# Label mapping
label_map = model.config.id2label if hasattr(model.config, "id2label") else {0: "NEGATIVE", 1: "POSITIVE"}

def classify_text(text):
    """
    Classifies the input text and returns the predicted label and confidence.
    
    Args:
        text (str): The input text to classify.

    Returns:
        dict: {
            "text": str,
            "prediction": str (e.g., "POSITIVE"),
            "confidence": float (e.g., 0.85),
            "quality": str (e.g., "OK", "GOOD", "BAD")
        }
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1)
    conf, pred = torch.max(probs, dim=1)
    confidence = conf.item()
    prediction = label_map.get(pred.item(), str(pred.item()))

    # Categorize confidence quality
    if confidence >= 0.8:
        quality = "OK"
    elif confidence >= 0.61:
        quality = "GOOD"
    elif confidence >= 0.5:
        quality = "LOW"
    else:
        quality = "BAD"

    return {
        "text": text,
        "prediction": prediction,
        "confidence": confidence,
        "quality": quality
    }
