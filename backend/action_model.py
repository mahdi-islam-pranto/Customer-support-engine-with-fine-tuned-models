"""
action_model.py — Loads your fine-tuned Qwen2.5-1.5B LoRA adapter from HF Hub
and generates structured JSON from actionable customer queries.
"""

import json
import logging
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

BASE_MODEL_ID  = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_ADAPTER_ID = "mahdi2020/qwen2.5-1.5B-ecommerce-lora-fine-tuned"

# Token budget for generated JSON — keep low for speed
MAX_NEW_TOKENS = 128


class ActionModel:
    def __init__(
        self,
        base_model_id: str = BASE_MODEL_ID,
        lora_adapter_id: str = LORA_ADAPTER_ID,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Qwen base model ({base_model_id}) on {self.device} ...")

        # ── Quantization (4-bit if GPU available, else full precision on CPU) ──
        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # CPU fallback — slow but functional for development / small demos
            logger.warning(
                "No GPU detected. Qwen will run on CPU — expect ~30s per response."
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )

        logger.info(f"Loading LoRA adapter from {lora_adapter_id} ...")
        self.model = PeftModel.from_pretrained(base_model, lora_adapter_id)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_id, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Pre-compute the EOS token id for <|im_end|>
        self._im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        logger.info("Action model ready.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_prompt(self, user_message: str) -> str:
        return (
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _extract_json(self, raw: str) -> dict | None:
        """Try to parse JSON from raw model output.  Handles minor noise."""
        raw = raw.strip()

        # Direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try to extract first {...} block
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return None

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, user_message: str) -> dict:
        """
        Returns:
            {
                "intent":  str,
                "params":  dict,       # all fields except intent
                "raw":     str,        # raw model output (for debugging)
                "success": bool
            }
        """
        prompt = self._build_prompt(user_message)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,            # greedy — deterministic JSON
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self._im_end_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        parsed = self._extract_json(raw)

        if parsed is None:
            logger.warning(f"Could not parse JSON from model output: {raw!r}")
            return {
                "intent":  "unknown",
                "params":  {},
                "raw":     raw,
                "success": False,
            }

        intent = parsed.pop("intent", "unknown")
        return {
            "intent":  intent,
            "params":  parsed,          # remaining keys are the action params
            "raw":     raw,
            "success": True,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_action_model: ActionModel | None = None


def get_action_model() -> ActionModel:
    global _action_model
    if _action_model is None:
        _action_model = ActionModel()
    return _action_model