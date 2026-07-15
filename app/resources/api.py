from pathlib import Path
from typing import Optional
from enum import Enum

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = Path(__file__).parent.parent / "model" / "sentiment_model.joblib"

app = FastAPI(
    title="E-Commerce Sentiment Analysis API",
    description="Analyse sentiment of product reviews and comments from e-commerce listings.",
    version="1.0.0",
)

model = joblib.load(MODEL_PATH)

CLASS_LABELS = {0: "negative", 1: "positive"}


class ProductCategory(str, Enum):
    electronics = "electronics"
    clothing = "clothing"
    home_and_garden = "home_and_garden"
    sports = "sports"
    beauty = "beauty"
    toys = "toys"
    books = "books"
    other = "other"


class ReviewRequest(BaseModel):
    comment: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The review text or comment left by the customer.",
        examples=["This product exceeded my expectations! Great build quality."],
    )
    product_id: Optional[str] = Field(
        None,
        description="Unique identifier for the product being reviewed.",
        examples=["PROD-12345"],
    )
    product_name: Optional[str] = Field(
        None,
        max_length=255,
        description="Name of the product.",
        examples=["Wireless Noise-Cancelling Headphones"],
    )
    category: Optional[ProductCategory] = Field(
        None,
        description="Product category to improve classification context.",
    )
    rating: Optional[float] = Field(
        None,
        ge=1.0,
        le=5.0,
        description="Star rating given by the customer (1.0 – 5.0).",
        examples=[4.5],
    )
    verified_purchase: Optional[bool] = Field(
        None,
        description="Whether the review is from a verified purchase.",
    )
    reviewer_id: Optional[str] = Field(
        None,
        description="Anonymised identifier for the reviewer.",
        examples=["USR-98765"],
    )


class SentimentResponse(BaseModel):
    sentiment: str = Field(description="Predicted sentiment: positive or negative.")
    confidence: float = Field(description="Model confidence score between 0 and 1.")
    comment_preview: str = Field(description="First 100 characters of the submitted comment.")


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}


@app.post("/analyse", response_model=SentimentResponse, tags=["Sentiment"])
def analyse_sentiment(review: ReviewRequest):
    text = review.comment

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    predicted_label = CLASS_LABELS[int(prediction)]
    confidence = round(float(probabilities[int(prediction)]), 4)

    return SentimentResponse(
        sentiment=predicted_label,
        confidence=confidence,
        comment_preview=text[:100],
    )
