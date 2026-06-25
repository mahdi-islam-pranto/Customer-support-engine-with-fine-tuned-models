"""
intent_router.py — Binary intent classifier using your fine-tuned DistilBERT.
Returns 0 (informative) or 1 (actionable).
Loads model once at startup and reuses it for every request.
"""

import logging
from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

HF_MODEL_ID = "mahdi2020/fine-tuned-distilbert-customer-intent-router"
LABEL_INFORMATIVE = 0
LABEL_ACTIONABLE  = 1


class IntentRouter:
    def __init__(self, model_id: str = HF_MODEL_ID):
        logger.info(f"Loading intent router from {model_id} ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Intent router ready on {self.device}")

    def predict(self, text: str) -> dict:
        """
        Returns:
            {
                "label": 0 | 1,
                "type":  "informative" | "actionable",
                "confidence": float  (0.0 – 1.0)
            }
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        label = int(torch.argmax(probs).item())
        confidence = float(probs[label].item())

        return {
            "label":      label,
            "type":       "actionable" if label == LABEL_ACTIONABLE else "informative",
            "confidence": round(confidence, 4),
        }


# ── Singleton — loaded once when the module is first imported ─────────────────

_router: IntentRouter | None = None


def get_router() -> IntentRouter:
    global _router
    if _router is None:
        _router = IntentRouter()
    return _router