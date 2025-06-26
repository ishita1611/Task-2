# Text Classifier with LangGraph

## 📦 Requirements
- Python 3.8+
- `transformers`, `datasets`, `peft`, `accelerate`, `langgraph`, `langchain-core`, `torch`

## 🏋️ Fine-tuning Instructions
1. Use HuggingFace's `Trainer` API or PEFT LoRA to fine-tune `distilbert-base-uncased` on IMDB or similar dataset.
2. Save with:
```python
model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")
