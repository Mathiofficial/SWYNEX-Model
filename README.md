# SWYNEX – Model or API Integration (Task 2)

**Support Ticket Triage**: a small prototype that integrates two pretrained
Hugging Face models to automatically triage customer support tickets.

## Problem
Support teams receive many tickets and must decide quickly *who* should handle
each one and *how urgent* it is. This prototype automates that first pass.

## How it works
1. **Sentiment analysis** (`distilbert-base-uncased-finetuned-sst-2-english`) – detects how negative the message is.
2. **Zero-shot classification** (`facebook/bart-large-mnli`) – routes the ticket to one of: billing, technical issue, account access, delivery, general inquiry.
3. **Priority rule** – NEGATIVE with confidence ≥ 0.90 → High, other NEGATIVE → Medium, otherwise Low.

No API keys are used or needed. Models are downloaded from the public Hugging Face Hub on first run.

## Setup
```bash
git clone https://github.com/<your-username>/SWYNEX-Model-or-API-Integration.git
cd SWYNEX-Model-or-API-Integration
pip install -r requirements.txt
```

## Usage
```bash
# Single ticket
python app.py --text "I was charged twice and nobody replies to me."

# Batch of tickets, saving JSON results
python app.py --file examples/sample_inputs.txt --out examples/sample_outputs.json
```

## Example
**Input**
```
I was charged twice for my subscription this month and nobody is replying to me.
```
**Output** (format; exact scores may vary slightly)
```
Sentiment: NEGATIVE (0.99)
Category : billing (0.9x)
Priority : High
```
Run the batch command above to generate `examples/sample_outputs.json` with real results for all five sample tickets.

## Project structure
```
app.py                    # triage logic + CLI
requirements.txt
examples/sample_inputs.txt
```

## Notes / Limitations
- Zero-shot classification is a baseline; fine-tuning on real ticket data would improve accuracy.
- The priority rule is intentionally simple and easy to extend.

## Tech
Python, Hugging Face Transformers, PyTorch
