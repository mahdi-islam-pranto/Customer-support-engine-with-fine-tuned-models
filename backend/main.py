"""
main.py — FastAPI backend orchestrating the full chatbot pipeline.

Startup:  loads all three models once (DeBERTa, FAISS, Qwen).
Endpoint: POST /chat   → { query: str } → { response, type, debug }
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from action_model  import get_action_model
from intent_router import get_router
from mock_api      import execute_action
from retriever     import get_retriever

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan: load all models at startup ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Loading models — this takes 1-3 minutes on first run ===")
    get_router()          # DeBERTa intent classifier
    get_retriever()       # FAISS + SentenceTransformer
    get_action_model()    # Qwen LoRA
    logger.info("=== All models loaded. API is ready. ===")
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="E-Commerce Customer Support Chatbot",
    description="Routes queries to FAISS (informative) or Qwen (actionable).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, example="Cancel my order #789456")


class ChatResponse(BaseModel):
    response:    str
    type:        str          # "informative" | "actionable" | "error"
    intent:      str | None   # populated for actionable queries
    success:     bool
    latency_ms:  int
    debug:       dict[str, Any] | None = None


# ── Main endpoint ─────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start = time.perf_counter()
    query = request.query.strip()

    try:
        # ── Step 1: classify intent ───────────────────────────────────────
        router     = get_router()
        routing    = router.predict(query)
        query_type = routing["type"]           # "informative" | "actionable"

        logger.info(
            f"[ROUTER] '{query[:60]}' → {query_type} "
            f"(confidence {routing['confidence']:.2%})"
        )

        # ── Step 2a: informative — retrieve from FAISS ────────────────────
        if query_type == "informative":
            retriever  = get_retriever()
            result     = retriever.search(query)

            latency_ms = int((time.perf_counter() - start) * 1000)
            return ChatResponse(
                response   = result["answer"],
                type       = "informative",
                intent     = None,
                success    = result["confident"],
                latency_ms = latency_ms,
                debug      = {
                    "matched_question": result["question"],
                    "faiss_distance":   result["distance"],
                    "router_confidence": routing["confidence"],
                },
            )

        # ── Step 2b: actionable — generate JSON with Qwen then call API ──
        action_model = get_action_model()
        generated    = action_model.generate(query)

        logger.info(
            f"[QWEN] intent={generated['intent']} "
            f"params={generated['params']} raw={generated['raw']!r}"
        )

        if not generated["success"]:
            latency_ms = int((time.perf_counter() - start) * 1000)
            return ChatResponse(
                response   = (
                    "I understood this is an action request but I couldn't "
                    "extract the necessary details. Could you rephrase? "
                    "For example: 'Cancel order #123456'."
                ),
                type       = "actionable",
                intent     = "unknown",
                success    = False,
                latency_ms = latency_ms,
                debug      = {"raw_output": generated["raw"]},
            )

        # ── Step 3: execute the mock API call ─────────────────────────────
        api_result = execute_action(
            intent = generated["intent"],
            params = generated["params"],
        )

        logger.info(f"[API] result={api_result}")

        latency_ms = int((time.perf_counter() - start) * 1000)
        return ChatResponse(
            response   = api_result["message"],
            type       = "actionable",
            intent     = generated["intent"],
            success    = api_result["success"],
            latency_ms = latency_ms,
            debug      = {
                "extracted_params":  generated["params"],
                "api_reference":     api_result.get("reference"),
                "router_confidence": routing["confidence"],
            },
        )

    except Exception as exc:
        logger.exception(f"Unhandled error for query: {query!r}")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Quick smoke-test: run with `python main.py` ───────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)