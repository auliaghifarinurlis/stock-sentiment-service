from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = FastAPI()

# Load model and tokenizer
MODEL_PATH = "./model.pt"
# TOKENIZER_PATH = "./tokenizer"

try:
    tokenizer = BertTokenizer.from_pretrained("indolem/indobert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("indolem/indobert-base-uncased", num_labels=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))  # Load weights
    model.eval()  # Set model to evaluation mode
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    print("Model and tokenizer successfully loaded.")
except Exception as e:
    print(f"Error loading model/tokenizer: {e}")
    exit()

# Input schema
class PredictRequest(BaseModel):
    text: str

# Prediction endpoint
@app.post("/predict")
def predict(req: PredictRequest):
    try:
        inputs = tokenizer(
            req.text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()
        label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = label_mapping.get(predicted_class, "Unknown")
        return {"text": req.text, "sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))