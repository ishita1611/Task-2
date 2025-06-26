import torch
import torch.nn.functional as F
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
import datetime
import os

# ==== SETUP ====
logging.basicConfig(
    filename="classifier.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./finetuned_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

label_map = model.config.id2label if hasattr(model.config, "id2label") else {0: "NEGATIVE", 1: "POSITIVE"}
CONFIDENCE_THRESHOLD = 0.75

def format_confidence_label(score):
    if score >= 0.80:
        return "✅ OK"
    elif score >= 0.61:
        return "👍 GOOD"
    elif score >= 0.50:
        return "🟡 OKAY"
    elif score >= 0.40:
        return "⚠️ BAD"
    else:
        return "❌ VERY BAD"

# ==== NODE FUNCTIONS ====

def inference_node(state):
    text = state.get("text", "").strip()
    if not text:
        raise ValueError("No input text provided.")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = F.softmax(logits, dim=1)
    confidence, pred = torch.max(probs, dim=1)
    label = label_map.get(pred.item(), str(pred.item()))

    logging.info(f"[InferenceNode] Input: {text} | Prediction: {label} | Confidence: {confidence.item():.2f}")

    return {
        "text": text,
        "prediction": label,
        "confidence": confidence.item()
    }

def confidence_check_node(state):
    if state["confidence"] >= CONFIDENCE_THRESHOLD:
        state["route"] = "accept"
    else:
        logging.info(f"[ConfidenceCheckNode] Confidence too low ({state['confidence']:.2f}). Triggering fallback.")
        state["route"] = "fallback"
    return state

def fallback_node(state):
    print(f"\n🤔 [FallbackNode] Not confident in prediction.")
    print(f"Text: {state['text']}")
    print(f"Prediction: {state['prediction']} (Confidence: {state['confidence']:.2f})")
    corrected = input(f"Could you clarify your intent? Was this a {state['prediction']} review? ").strip()

    state["prediction"] = corrected

    # Save feedback
    with open("user_feedback.tsv", "a") as f:
        f.write(f"{state['text']}\t{corrected}\t{state['confidence']:.2f}\n")

    logging.info(f"[FallbackNode] User clarified as: {corrected}")
    return state

def final_output(state):
    conf_label = format_confidence_label(state["confidence"])
    print(f"\n🎯 Final Label: {state['prediction']} ({conf_label}, {state['confidence']:.2f})")
    logging.info(f"[FinalOutput] Final: {state['prediction']} ({state['confidence']:.2f})")
    return state

# ==== GRAPH SETUP ====

graph = StateGraph(dict)

graph.add_node("inference", RunnableLambda(inference_node))
graph.add_node("confidence_check", RunnableLambda(confidence_check_node))
graph.add_node("fallback", RunnableLambda(fallback_node))
graph.add_node("final", RunnableLambda(final_output))

graph.set_entry_point("inference")
graph.add_edge("inference", "confidence_check")
graph.add_conditional_edges("confidence_check", {
    "accept": RunnableLambda(final_output),
    "fallback": RunnableLambda(fallback_node)
})
graph.add_edge("fallback", "final")
graph.set_finish_point("final")

app = graph.compile()

# ==== MAIN LOOP ====
try:
    while True:
        user_input = input("\n📥 Input: ").strip()
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        result = app.invoke({"text": user_input})
except KeyboardInterrupt:
    print("\n👑 Session Interrupted.")
