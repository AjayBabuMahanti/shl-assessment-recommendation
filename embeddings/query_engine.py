# ============================================================
# SHL Assessment Query Engine  — Friendly RAG Bot
# (Mistral AI  +  ChromaDB  +  SentenceTransformer)
# ============================================================
#
# Usage:
#   python embeddings/query_engine.py                    → interactive chat
#   python embeddings/query_engine.py "your query here"  → one-shot answer
#   from embeddings.query_engine import query_shl        → library import
# ============================================================

import os
import re
import sys
import subprocess
import textwrap
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer
from mistralai import Mistral

# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────

MISTRAL_API_KEY   = "ZEI8aC50vc9bjZToUkSuH15wSZJU4RPY"
MISTRAL_MODEL     = "mistral-small-latest"

EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
COLLECTION_NAME   = "shl_assessments"
TOP_K             = 10
DISPLAYED_RESULTS = 5
MAX_TOKENS        = 1200

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DB  = os.path.join(BASE_DIR, "vector_db")

# ──────────────────────────────────────────────────────────
# Lazy singletons
# ──────────────────────────────────────────────────────────

_embed_model:    Optional[SentenceTransformer] = None
_chroma_col                                    = None
_mistral_client: Optional[Mistral]             = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def _get_collection():
    """
    Connect to ChromaDB and return the collection.
    If the collection is missing, offers to auto-build the index.
    """
    global _chroma_col
    if _chroma_col is not None:
        return _chroma_col

    client = chromadb.PersistentClient(path=VECTOR_DB)

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME not in existing:
        raise _CollectionMissing(
            f"The assessment index '{COLLECTION_NAME}' hasn't been built yet.\n"
            "Run:  python embeddings/create_embeddings.py"
        )

    _chroma_col = client.get_collection(COLLECTION_NAME)
    return _chroma_col


class _CollectionMissing(Exception):
    pass


def _get_mistral() -> Mistral:
    global _mistral_client
    if _mistral_client is None:
        _mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    return _mistral_client


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _is_assessment_query(text: str) -> bool:
    """
    Rough heuristic: does the user seem to be asking about
    a job/skill/assessment topic?
    """
    keywords = [
        "job", "role", "skill", "test", "assessment", "hire", "hiring",
        "developer", "engineer", "analyst", "manager", "sales", "data",
        "python", "java", "sql", "leadership", "personality", "cognitive",
        "aptitude", "simulation", "programming", "software", "finance",
        "recommend", "find", "suggest", "looking for", "need", "want",
        "score", "candidate", "recruitment", "position", "competency",
    ]
    low = text.lower()
    return any(kw in low for kw in keywords)


def _retrieve(query: str, n_results: int = TOP_K) -> list[dict]:
    col = _get_collection()
    vec = _get_embed_model().encode([query]).tolist()
    # Clamp n_results to at least 1
    n   = max(1, n_results)
    res = col.query(
        query_embeddings=vec,
        n_results=n,
        include=["metadatas", "distances"],
    )
    out = []
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        out.append({**meta, "_distance": dist})
    return out


def _build_context(candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates, 1):
        desc = c.get("description", "N/A")
        if len(desc) > 300:
            desc = desc[:297] + "…"
        lines.append(
            f"[{i}] {c.get('name','N/A')}\n"
            f"    URL      : {c.get('url','N/A')}\n"
            f"    Type     : {c.get('test_type','N/A')}\n"
            f"    Duration : {c.get('duration','N/A')}\n"
            f"    Adaptive : {c.get('adaptive_support','N/A')}  |  "
            f"Remote: {c.get('remote_support','N/A')}\n"
            f"    Desc     : {desc}\n"
        )
    return "\n".join(lines)


def _ask_mistral_recommend(query: str, context: str) -> str:
    """Call Mistral to produce friendly, ranked assessment recommendations."""
    system = textwrap.dedent(f"""
        You are "SHL Scout" — a warm, knowledgeable, and friendly talent assessment
        advisor. You speak like a helpful friend who works at SHL, not like a robot.

        When a user describes a job or skill, you:
        1. Acknowledge their request naturally (1 short sentence).
        2. Recommend the top {DISPLAYED_RESULTS} most suitable assessments from the
           numbered list provided, ranked best-to-worst fit.
        3. For each, give:
           • **Assessment name** (bold)
           • A 1–2 sentence *why it fits* explanation — be specific & friendly
           • Test type, duration (mention if N/A), and remote/adaptive status
           • The URL
        4. End with an encouraging closing line.

        Be concise, warm, and professional. Use bullet points.
        Only recommend assessments from the provided list.
    """).strip()

    user_msg = (
        f"I'm looking for SHL assessments for: **{query}**\n\n"
        f"Here are the candidate assessments retrieved:\n\n{context}"
    )

    resp = _get_mistral().chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


def _ask_mistral_general(query: str) -> str:
    """
    Handle off-topic or general questions with Mistral in a friendly way,
    and gently steer the user back to SHL assessments.
    """
    system = textwrap.dedent("""
        You are "SHL Scout" — a friendly AI assistant specialised in SHL talent
        assessments. You are warm, concise, and approachable.

        If the user asks something not directly related to jobs or SHL assessments:
        - Answer their question briefly and kindly.
        - Then gently remind them what you *can* help with: finding the right SHL
          assessment for any job role, skill set, or hiring scenario.
        - Invite them to ask a job-related query.

        Never be rude or dismissive. Keep it short and upbeat.
    """).strip()

    resp = _get_mistral().chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": query},
        ],
        max_tokens=400,
        temperature=0.5,
    )
    return resp.choices[0].message.content.strip()


# ──────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────

def query_shl(
    user_query: str,
    top_k:      int  = TOP_K,
    n_show:     int  = DISPLAYED_RESULTS,
    verbose:    bool = False,
) -> dict:
    """
    End-to-end RAG query. Returns a dict with keys:
        query, candidates, answer
    """
    candidates: list[dict] = []
    answer: str = ""

    try:
        candidates = _retrieve(user_query)
    except _CollectionMissing:
        # Index not built yet — offer to build it
        answer = (
            "😅 Heads up! The assessment database hasn't been indexed yet.\n\n"
            "No worries — I'll build it for you right now! This takes about "
            "1–2 minutes the first time.\n\n"
            "Running: python embeddings/create_embeddings.py …"
        )
        print(answer)
        _build_index()
        # retry
        candidates = _retrieve(user_query)

    if verbose:
        print("\n── Retrieved candidates ───────────────────────────────")
        for i, c in enumerate(candidates, 1):
            print(f"  {i}. {c.get('name')} (dist={c['_distance']:.4f})")
        print("───────────────────────────────────────────────────────\n")

    if not candidates:
        answer = (
            "Hmm, I couldn't find anything matching that in the database. "
            "Try describing the job role or skills differently — I'm here to help! 😊"
        )
    elif _is_assessment_query(user_query):
        context = _build_context(candidates)
        answer  = _ask_mistral_recommend(user_query, context)
    else:
        # Off-topic query — use general friendly handler
        answer = _ask_mistral_general(user_query)

    return {"query": user_query, "candidates": candidates, "answer": answer}


def _build_index() -> None:
    """Run create_embeddings.py as a subprocess to build the vector DB."""
    script = os.path.join(BASE_DIR, "embeddings", "create_embeddings.py")
    py     = sys.executable
    result = subprocess.run([py, script], capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(
            "Index build failed. Please run:\n"
            "  python embeddings/create_embeddings.py"
        )
    global _chroma_col
    _chroma_col = None   # force reconnect


# ──────────────────────────────────────────────────────────
# Interactive CLI
# ──────────────────────────────────────────────────────────

BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║     👋  Hi! I'm SHL Scout — your assessment advisor!     ║
║  Tell me about a job role or the skills you're looking   ║
║  for, and I'll recommend the best SHL assessments!       ║
║                                                          ║
║  Commands:  'quit'/'exit' → leave  |  'verbose' → debug  ║
╚══════════════════════════════════════════════════════════╝
"""

EXAMPLES = [
    "Looking for a Python developer with SQL skills",
    "Hiring a sales manager for a retail chain",
    "Entry-level data science analyst role",
]


def _sep(char: str = "─", w: int = 60) -> None:
    print(char * w)


def _interactive_loop() -> None:
    print(BANNER)
    print("💡 Example queries:")
    for ex in EXAMPLES:
        print(f"   → {ex}")
    _sep()

    verbose = False

    while True:
        try:
            raw = input("\n🤔  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋  Take care! Come back anytime. Goodbye!")
            break

        if not raw:
            print("   (I'm listening… just type your query and hit Enter!)")
            continue

        low = raw.lower()

        if low in {"quit", "exit", "q", "bye", "goodbye"}:
            print("👋  Great chatting with you! Goodbye!")
            break

        if low == "verbose":
            verbose = not verbose
            state = "ON 🔍" if verbose else "OFF"
            print(f"🔧  Debug / verbose mode is now {state}")
            continue

        try:
            print("⏳  Let me look that up for you…\n")
            result = query_shl(raw, verbose=verbose)
            _sep("═")
            print("\n🎯  SHL Scout says:\n")
            print(result["answer"])
            _sep()

        except KeyboardInterrupt:
            print("\n\n👋  Goodbye!")
            break
        except Exception as exc:  # noqa: BLE001
            print(
                f"\n😬  Uh oh, something went wrong on my end:\n   {exc}\n\n"
                "Please try again or re-run the embeddings script if the "
                "database seems corrupted."
            )


# ──────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # One-shot mode: python query_engine.py "your query"
        q      = " ".join(sys.argv[1:])
        print(f"\n🔍  Running one-shot query: \"{q}\"\n")
        result = query_shl(q, verbose=True)
        _sep("═")
        print("\n🎯  SHL Scout says:\n")
        print(result["answer"])
        _sep("═")
    else:
        _interactive_loop()