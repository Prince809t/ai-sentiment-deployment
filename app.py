from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline


# ============================================
# APP
# ============================================

app = FastAPI(
    title="AI Movie Review Sentiment API",
    description="Sentiment classification using a fine-tuned DistilBERT model."
)


# ============================================
# MODEL
# ============================================

MODEL_ID = "princesolanki497/imdb-sentiment-distilbert-v2"

print("Loading model...")

classifier = pipeline(
    "text-classification",
    model=MODEL_ID,
    top_k=1
)

print("Model loaded successfully!")


# ============================================
# REQUEST FORMAT
# ============================================

class ReviewRequest(BaseModel):

    text: str


# ============================================
# HOME
# ============================================

@app.get("/")
def home():

    return {
        "message": "AI Movie Review Sentiment API is running",
        "model": MODEL_ID
    }


# ============================================
# PREDICT
# ============================================

@app.post("/predict")
def predict_sentiment(request: ReviewRequest):

    text = request.text.strip()

    if not text:

        return {
            "error": "Please enter a review."
        }

    result = classifier(text)

    if isinstance(result[0], list):

        prediction = result[0][0]

    else:

        prediction = result[0]

    return {
        "text": text,
        "label": prediction["label"],
        "score": prediction["score"]
    }