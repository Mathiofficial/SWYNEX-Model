"""Support Ticket Triage - SWYNEX Task 2 (Model or API Integration).

Uses two pretrained Hugging Face models through the `transformers` library:
  1. Sentiment analysis  -> how upset is the customer?
  2. Zero-shot classification -> which team should handle the ticket?
A simple rule then combines both into a priority level.

No API keys are required. Models are downloaded once from the public
Hugging Face Hub and then run locally.
"""
import argparse
import json
import sys

from transformers import pipeline

SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
CATEGORIES = ["billing", "technical issue", "account access", "delivery", "general inquiry"]


def load_models():
    print("Loading models (first run downloads them)...", file=sys.stderr)
    sentiment = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
    classifier = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)
    return sentiment, classifier


def priority_from(label: str, score: float) -> str:
    if label == "NEGATIVE" and score >= 0.90:
        return "High"
    if label == "NEGATIVE":
        return "Medium"
    return "Low"


def triage(text: str, sentiment, classifier) -> dict:
    sent = sentiment(text[:512])[0]
    cls = classifier(text, candidate_labels=CATEGORIES)
    return {
        "input": text,
        "sentiment": sent["label"],
        "sentiment_score": round(sent["score"], 3),
        "category": cls["labels"][0],
        "category_score": round(cls["scores"][0], 3),
        "priority": priority_from(sent["label"], sent["score"]),
    }


def main():
    parser = argparse.ArgumentParser(description="AI support ticket triage")
    parser.add_argument("--text", help="A single ticket text")
    parser.add_argument("--file", help="Text file with one ticket per line")
    parser.add_argument("--out", help="Write results to this JSON file")
    args = parser.parse_args()

    if args.text:
        tickets = [args.text]
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            tickets = [line.strip() for line in f if line.strip()]
    else:
        parser.error("Provide --text or --file")

    sentiment, classifier = load_models()
    results = [triage(t, sentiment, classifier) for t in tickets]

    for r in results:
        print(f"\nTicket   : {r['input']}")
        print(f"Sentiment: {r['sentiment']} ({r['sentiment_score']})")
        print(f"Category : {r['category']} ({r['category_score']})")
        print(f"Priority : {r['priority']}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
