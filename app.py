"""
app.py — Streamlit customer support chatbot.
Self-contained: loads DeBERTa, FAISS, and Qwen directly (no API).
Includes a live mock database panel that updates as actions are taken.
"""

import json
import pickle
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import faiss
import streamlit as st
import torch

# ── Page config — must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="Shop Support",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
FAISS_INDEX = BASE_DIR / "models" / "faiss" / "faq_index.faiss"
METADATA    = BASE_DIR / "models" / "faiss" / "faq_metadata.pkl"

HF_DEBERTA  = "mahdi2020/fine-tuned-distilbert-customer-intent-router"
HF_BASE     = "Qwen/Qwen2.5-1.5B-Instruct"
HF_LORA     = "mahdi2020/qwen2.5-1.5B-ecommerce-lora-fine-tuned"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ─────────────────────────────────────────────────────────────────────────────
# MOCK DATABASE
# ─────────────────────────────────────────────────────────────────────────────

def _initial_db() -> dict:
    now = datetime.now()
    return {
        "orders": {
            "764283": {"status": "shipped",    "product": "Wireless Headphones",  "address": "10 Old St",      "eta": (now + timedelta(days=2)).strftime("%d %b %Y"), "customer": "alice@example.com"},
            "243565": {"status": "processing", "product": "Running Shoes (UK 9)", "address": "22 Baker Ave",   "eta": (now + timedelta(days=4)).strftime("%d %b %Y"), "customer": "bob@example.com"},
            "750175": {"status": "processing", "product": "Coffee Maker",         "address": "5 Maple Rd",     "eta": (now + timedelta(days=5)).strftime("%d %b %Y"), "customer": "carol@example.com"},
            "789456": {"status": "processing", "product": "Gaming Mouse",         "address": "88 Pine Blvd",   "eta": (now + timedelta(days=3)).strftime("%d %b %Y"), "customer": "dave@example.com"},
            "334455": {"status": "shipped",    "product": "Mechanical Keyboard",  "address": "14 Oak Lane",    "eta": (now + timedelta(days=1)).strftime("%d %b %Y"), "customer": "eve@example.com"},
            "998877": {"status": "delivered",  "product": "USB-C Hub",            "address": "31 Elm Street",  "eta": "Delivered",                                     "customer": "frank@example.com"},
            "112233": {"status": "processing", "product": "Yoga Mat",             "address": "7 Birch Close",  "eta": (now + timedelta(days=6)).strftime("%d %b %Y"), "customer": "alice@example.com"},
        },
        "accounts": {
            "alice@example.com": {"name": "Alice Chen",    "plan": "premium", "active": True},
            "bob@example.com":   {"name": "Bob Martin",    "plan": "basic",   "active": True},
            "carol@example.com": {"name": "Carol White",   "plan": "platinum","active": True},
            "dave@example.com":  {"name": "Dave Kumar",    "plan": "basic",   "active": True},
            "eve@example.com":   {"name": "Eve Torres",    "plan": "premium", "active": True},
            "frank@example.com": {"name": "Frank Lee",     "plan": "basic",   "active": True},
        },
        "activity_log": [],
    }

def _log_activity(db: dict, action: str, details: str):
    db["activity_log"].insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "action": action,
        "details": details,
    })
    if len(db["activity_log"]) > 20:
        db["activity_log"] = db["activity_log"][:20]

def execute_action(db: dict, intent: str, params: dict) -> str:
    """Mutates db in place and returns a human-readable response string."""
    orders   = db["orders"]
    accounts = db["accounts"]

    if intent == "cancel_order":
        oid = params.get("order_id", "")
        if oid not in orders:
            return f"❌ Order #{oid} not found."
        if orders[oid]["status"] not in ("processing", "pending"):
            return f"❌ Order #{oid} is already **{orders[oid]['status']}** and cannot be cancelled."
        orders[oid]["status"] = "cancelled"
        _log_activity(db, "cancel_order", f"Order #{oid} → cancelled")
        return f"✅ Order #{oid} ({orders[oid]['product']}) has been **cancelled**. Refund in 3–5 business days."

    elif intent == "set_up_shipping_address":
        oid     = params.get("order_id", "")
        new_addr = params.get("new_address", "")
        if oid not in orders:
            return f"❌ Order #{oid} not found."
        if orders[oid]["status"] not in ("processing", "pending"):
            return f"❌ Cannot update address — order #{oid} is already {orders[oid]['status']}."
        old = orders[oid]["address"]
        orders[oid]["address"] = new_addr
        _log_activity(db, "update_address", f"Order #{oid}: '{old}' → '{new_addr}'")
        return f"✅ Delivery address updated for order #{oid}.\n- **Old:** {old}\n- **New:** {new_addr}"

    elif intent == "change_order":
        oid = params.get("order_id", "")
        if oid not in orders:
            return f"❌ Order #{oid} not found."
        if orders[oid]["status"] not in ("processing", "pending"):
            return f"❌ Order #{oid} is {orders[oid]['status']} — too late to modify."
        _log_activity(db, "change_order", f"Order #{oid} flagged for modification")
        return f"✅ Order #{oid} ({orders[oid]['product']}) is open for changes. Please specify what you'd like to modify."

    elif intent == "track_order":
        oid = params.get("order_id", "")
        if oid not in orders:
            return f"❌ Order #{oid} not found."
        o = orders[oid]
        return (
            f"📦 **Order #{oid} — {o['product']}**\n"
            f"- Status: **{o['status'].capitalize()}**\n"
            f"- Address: {o['address']}\n"
            f"- ETA: {o['eta']}"
        )

    elif intent == "return_item":
        oid = params.get("order_id", "")
        if oid not in orders:
            return f"❌ Order #{oid} not found."
        if orders[oid]["status"] != "delivered":
            return f"❌ Order #{oid} has not been delivered yet."
        orders[oid]["status"] = "return_requested"
        _log_activity(db, "return_item", f"Order #{oid} return requested")
        return f"✅ Return initiated for order #{oid} ({orders[oid]['product']}). A prepaid label will be emailed within 24 hours."

    elif intent == "delete_account":
        email = params.get("email", "")
        if email not in accounts:
            return f"✅ If an account for **{email}** exists, it has been scheduled for deletion."
        accounts[email]["active"] = False
        _log_activity(db, "delete_account", f"Account {email} deactivated")
        return f"✅ Account for **{accounts[email]['name']}** ({email}) scheduled for deletion within 30 days."

    elif intent == "recover_password":
        email = params.get("email", "")
        _log_activity(db, "recover_password", f"Password reset sent to {email}")
        return f"✅ A password reset link has been sent to **{email}** (expires in 15 min)."

    elif intent == "get_invoice":
        oid = params.get("order_id", "")
        if oid not in orders:
            return f"❌ Order #{oid} not found."
        _log_activity(db, "get_invoice", f"Invoice sent for order #{oid}")
        return f"✅ Invoice for order #{oid} ({orders[oid]['product']}) has been emailed to the customer."

    else:
        return f"⚠️ I understood the intent **{intent}** but don't have a handler for it yet."


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (cached — loads once per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_intent_router():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok   = AutoTokenizer.from_pretrained(HF_DEBERTA)
    model = AutoModelForSequenceClassification.from_pretrained(HF_DEBERTA)
    model.eval()
    return tok, model

@st.cache_resource(show_spinner=False)
def load_retriever():
    from sentence_transformers import SentenceTransformer
    index = faiss.read_index(str(FAISS_INDEX))
    with open(METADATA, "rb") as f:
        raw = pickle.load(f)
    if isinstance(raw[0], str):
        metadata = [{"question": "", "answer": a} for a in raw]
    else:
        metadata = raw
    embedder = SentenceTransformer(EMBED_MODEL)
    return index, metadata, embedder

@st.cache_resource(show_spinner=False)
def load_action_model():
    from peft import PeftModel
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                               BitsAndBytesConfig)
    tok = AutoTokenizer.from_pretrained(HF_BASE, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        cfg   = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16)
        base  = AutoModelForCausalLM.from_pretrained(HF_BASE, quantization_config=cfg,
                                                     device_map="auto", trust_remote_code=True)
    else:
        base  = AutoModelForCausalLM.from_pretrained(HF_BASE, torch_dtype=torch.float32,
                                                     device_map="cpu", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, HF_LORA)
    model.eval()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    return tok, model, im_end


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def classify_intent(query: str):
    tok, model = load_intent_router()
    inputs = tok(query, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    label = int(torch.argmax(probs).item())
    return label, float(probs[label].item())

def retrieve_answer(query: str):
    index, metadata, embedder = load_retriever()
    vec = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    distances, indices = index.search(vec, 1)
    dist = float(distances[0][0])
    idx  = int(indices[0][0])
    if idx == -1 or dist > 0.8:
        return None, dist
    return metadata[idx].get("answer", "No answer available."), dist

def generate_action_json(query: str):
    import re
    tok, model, im_end = load_action_model()
    prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                             pad_token_id=tok.eos_token_id, eos_token_id=im_end)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    raw = tok.decode(new_tokens, skip_special_tokens=True).strip()
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group()), raw
            except json.JSONDecodeError:
                pass
    return None, raw


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "db" not in st.session_state:
    st.session_state.db = _initial_db()
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0f1117;
    color: #e8eaf0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] { background: #161a24; border-right: 1px solid #1e2535; }
[data-testid="stSidebar"] * { color: #c5c9d6 !important; }

/* ── Chat bubbles ── */
.bubble-wrap { display: flex; margin: 6px 0; }
.bubble-wrap.user  { justify-content: flex-end; }
.bubble-wrap.bot   { justify-content: flex-start; }
.bubble {
    max-width: 75%;
    padding: 11px 15px;
    border-radius: 16px;
    font-size: 0.92rem;
    line-height: 1.55;
    word-break: break-word;
}
.bubble.user {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: #fff;
    border-bottom-right-radius: 4px;
}
.bubble.bot {
    background: #1e2535;
    color: #dde1ed;
    border-bottom-left-radius: 4px;
    border: 1px solid #2a3045;
}
.bubble.bot.error { border-color: #ef4444; background: #1f1520; }
.meta { font-size: 0.72rem; color: #5a6380; margin: 2px 6px 8px; }
.meta.user { text-align: right; }

/* ── Tag badges ── */
.tag {
    display: inline-block;
    font-size: 0.68rem;
    padding: 2px 7px;
    border-radius: 20px;
    margin-right: 5px;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.tag-info   { background: #0e3a5c; color: #60c8f5; }
.tag-action { background: #1a3a1a; color: #4ade80; }
.tag-error  { background: #3a1a1a; color: #f87171; }
.tag-conf   { background: #2a2520; color: #fbbf24; }

/* ── DB panel ── */
.db-header {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7280;
    margin: 14px 0 6px;
}
.db-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 10px;
    border-radius: 8px;
    margin: 3px 0;
    background: #1a1f2e;
    font-size: 0.78rem;
}
.db-row:hover { background: #1e2535; }
.status-pill {
    font-size: 0.68rem;
    padding: 2px 8px;
    border-radius: 12px;
    font-weight: 600;
}
.s-processing { background: #1e3a5f; color: #60a5fa; }
.s-shipped    { background: #1a3a4a; color: #34d399; }
.s-delivered  { background: #1a3a1a; color: #4ade80; }
.s-cancelled  { background: #3a1a1a; color: #f87171; }
.s-return_requested { background: #3a2a1a; color: #fb923c; }
.s-other      { background: #2a2535; color: #c084fc; }

.log-entry {
    padding: 5px 8px;
    border-left: 2px solid #2563eb;
    margin: 4px 0;
    font-size: 0.75rem;
    background: #161a24;
    border-radius: 0 6px 6px 0;
}
.log-time { color: #6b7280; margin-right: 6px; }
.log-action { color: #60a5fa; font-weight: 600; }

/* ── Input area ── */
[data-testid="stChatInput"] > div {
    background: #1a1f2e !important;
    border: 1px solid #2a3045 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea { color: #e8eaf0 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #1a1f2e;
    border: 1px solid #2a3045;
    border-radius: 10px;
    padding: 10px 14px;
}

/* ── Scrollable db panel ── */
.db-scroll { max-height: 340px; overflow-y: auto; padding-right: 4px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — model loader + DB panel
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛍️ Shop Support")
    st.caption("AI-powered customer support engine")
    st.divider()

    # ── Model loading ─────────────────────────────────────────────────────
    if not st.session_state.models_loaded:
        st.markdown("### ⚙️ Load Models")
        st.caption("Models load once and stay cached for the session.")

        if st.button("🚀 Load All Models", use_container_width=True, type="primary"):
            steps = [
                ("🔍 Intent Router (DistilBERT)", load_intent_router),
                ("📚 FAQ Retriever (FAISS + MiniLM)", load_retriever),
                ("🤖 Action Model (Qwen2.5 + LoRA)", load_action_model),
            ]
            progress = st.progress(0, text="Starting...")
            status   = st.empty()
            all_ok   = True

            for i, (label, loader_fn) in enumerate(steps):
                status.markdown(f"**Loading:** {label}")
                try:
                    loader_fn()
                    progress.progress((i + 1) / len(steps), text=f"✅ {label}")
                    time.sleep(0.3)
                except Exception as e:
                    st.error(f"Failed to load {label}: {e}")
                    all_ok = False
                    break

            if all_ok:
                st.session_state.models_loaded = True
                status.empty()
                progress.empty()
                st.success("All models ready!", icon="✅")
                st.rerun()
    else:
        st.markdown("### ✅ Models Active")
        st.markdown("""
<div style="font-size:0.8rem; line-height:2;">
🔍 DistilBERT intent router<br>
📚 FAISS + MiniLM retriever<br>
🤖 Qwen2.5-1.5B + LoRA action model
</div>
""", unsafe_allow_html=True)
        device = "GPU 🟢" if torch.cuda.is_available() else "CPU 🟡"
        st.caption(f"Running on: **{device}**")

    st.divider()

    # ── Live Database Panel ───────────────────────────────────────────────
    st.markdown("### 🗄️ Live Database")

    db = st.session_state.db
    orders   = db["orders"]
    accounts = db["accounts"]
    log      = db["activity_log"]

    # Summary metrics
    c1, c2 = st.columns(2)
    with c1:
        processing = sum(1 for o in orders.values() if o["status"] == "processing")
        st.metric("Processing", processing)
    with c2:
        cancelled = sum(1 for o in orders.values() if o["status"] == "cancelled")
        st.metric("Cancelled", cancelled)

    # Orders table
    st.markdown('<div class="db-header">Orders</div>', unsafe_allow_html=True)
    st.markdown('<div class="db-scroll">', unsafe_allow_html=True)
    for oid, o in orders.items():
        s = o["status"]
        css = f"s-{s}" if f"s-{s}" in ["s-processing","s-shipped","s-delivered","s-cancelled","s-return_requested"] else "s-other"
        st.markdown(f"""
<div class="db-row">
  <span>#{oid}<br><span style="color:#6b7280;font-size:0.7rem">{o['product'][:22]}</span></span>
  <span class="status-pill {css}">{s}</span>
</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Accounts table
    st.markdown('<div class="db-header">Accounts</div>', unsafe_allow_html=True)
    st.markdown('<div class="db-scroll">', unsafe_allow_html=True)
    for email, acc in accounts.items():
        dot = "🟢" if acc["active"] else "🔴"
        st.markdown(f"""
<div class="db-row">
  <span>{dot} {acc['name']}<br>
  <span style="color:#6b7280;font-size:0.7rem">{email}</span></span>
  <span style="color:#9ca3af;font-size:0.72rem">{acc['plan']}</span>
</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Activity log
    if log:
        st.markdown('<div class="db-header">Recent Activity</div>', unsafe_allow_html=True)
        for entry in log[:6]:
            st.markdown(f"""
<div class="log-entry">
  <span class="log-time">{entry['time']}</span>
  <span class="log-action">{entry['action']}</span><br>
  <span style="color:#9ca3af">{entry['details']}</span>
</div>""", unsafe_allow_html=True)

    if st.button("🔄 Reset Database", use_container_width=True):
        st.session_state.db = _initial_db()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — chat interface
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## 💬 Customer Support")

# Example queries
if not st.session_state.messages:
    st.markdown("**Try an example query:**")
    examples = [
        "What is your return policy?",
        "Cancel my order #750175",
        "Change delivery address to 42 Wallaby Way for order #243565",
        "Track my order #789456",
        "What payment methods do you accept?",
        "Delete my account, email is carol@example.com",
    ]
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        if cols[i % 3].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state._pending_query = ex
            st.rerun()

st.divider()

# Render chat history
for msg in st.session_state.messages:
    role = msg["role"]
    wrap_class = "user" if role == "user" else "bot"
    bubble_class = f"bubble {wrap_class}"
    if role == "assistant" and msg.get("error"):
        bubble_class += " error"

    st.markdown(f'<div class="bubble-wrap {wrap_class}">', unsafe_allow_html=True)
    st.markdown(f'<div class="{bubble_class}">{msg["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if role == "assistant":
        tags_html = ""
        if "query_type" in msg:
            tag_class = "tag-info" if msg["query_type"] == "informative" else "tag-action"
            tags_html += f'<span class="tag {tag_class}">{msg["query_type"].upper()}</span>'
        if "confidence" in msg:
            tags_html += f'<span class="tag tag-conf">conf {msg["confidence"]:.0%}</span>'
        if "intent" in msg:
            tags_html += f'<span class="tag tag-action">{msg["intent"]}</span>'
        if tags_html:
            st.markdown(f'<div class="meta">{tags_html}</div>', unsafe_allow_html=True)


# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "Ask a question or request an action...",
    disabled=not st.session_state.models_loaded,
)

# Handle example button clicks
if hasattr(st.session_state, "_pending_query"):
    user_input = st.session_state._pending_query
    del st.session_state._pending_query

if not st.session_state.models_loaded and not user_input:
    st.info("👈 Click **Load All Models** in the sidebar to start the chatbot.", icon="ℹ️")

if user_input:
    if not st.session_state.models_loaded:
        st.warning("Please load the models first using the sidebar.", icon="⚠️")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f'<div class="bubble-wrap user"><div class="bubble user">{user_input}</div></div>', unsafe_allow_html=True)

    # Processing
    with st.spinner("Thinking..."):
        try:
            # Step 1: classify
            label, confidence = classify_intent(user_input)
            query_type = "actionable" if label == 1 else "informative"

            if query_type == "informative":
                # Step 2a: FAISS retrieval
                answer, dist = retrieve_answer(user_input)
                if answer is None:
                    response = ("I'm sorry, I couldn't find a specific answer to that. "
                                "Please contact our support team for further help.")
                    error = True
                else:
                    response = answer
                    error = False

                bot_msg = {
                    "role": "assistant",
                    "content": response,
                    "query_type": "informative",
                    "confidence": confidence,
                    "error": error,
                }

            else:
                # Step 2b: Qwen → JSON → execute
                parsed, raw = generate_action_json(user_input)

                if parsed is None:
                    response = ("I understood this is an action request but couldn't extract "
                                "the details. Could you rephrase? E.g. *'Cancel order #123456'*.")
                    bot_msg = {
                        "role": "assistant", "content": response,
                        "query_type": "actionable", "confidence": confidence, "error": True,
                    }
                else:
                    intent = parsed.pop("intent", "unknown")
                    params = parsed
                    response = execute_action(st.session_state.db, intent, params)
                    bot_msg = {
                        "role": "assistant",
                        "content": response,
                        "query_type": "actionable",
                        "confidence": confidence,
                        "intent": intent,
                        "error": False,
                    }

        except Exception as e:
            bot_msg = {
                "role": "assistant",
                "content": f"⚠️ An error occurred: {e}",
                "error": True,
            }

    st.session_state.messages.append(bot_msg)
    st.rerun()