"""
SHL Assessment Recommendation System — FastAPI Application
==========================================================

Endpoints:
  GET  /health     → Health check
  POST /recommend  → Get SHL assessment recommendations for a query

Run locally:
  uvicorn api.main:app --reload
  fastapi dev api/main.py
"""

import logging
import os
import sys
from typing import Annotated

import uvicorn
from fastapi import Body, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# ── Add project root to sys.path ──────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from embeddings.query_engine import _retrieve  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("shl_api")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MAX_RESULTS = 10

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────────


class RecommendRequest(BaseModel):
    """Request body for the /recommend endpoint."""

    query: str = Field(
        min_length=1,
        max_length=1000,
        description="Job role, skills, or hiring requirement description.",
        examples=["Looking for a Python developer with SQL and teamwork skills"],
    )


class AssessmentItem(BaseModel):
    """A single assessment recommendation."""

    assessment_name: str
    url: str
    test_type: str


class RecommendResponse(BaseModel):
    """Response body for the /recommend endpoint."""

    query: str
    total_recommendations: int
    recommendations: list[AssessmentItem]
    message: str = ""   # friendly message for off-topic or empty results


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────────────


# Keywords that indicate a hiring/assessment-related query
_JOB_KEYWORDS = [
    "job", "role", "skill", "test", "assess", "hire", "hiring", "recruit",
    "developer", "engineer", "analyst", "manager", "sales", "data", "nurse",
    "python", "java", "sql", "leadership", "personality", "cognitive", "behavior",
    "aptitude", "simulation", "programming", "software", "finance", "marketing",
    "recommend", "find", "suggest", "looking for", "need", "want", "candidate",
    "position", "competency", "experience", "graduate", "entry level", "work",
    "team", "communication", "customer", "service", "accounting", "clinical",
    "executive", "intern", "professional", "technical", "mechanical", "electrical",
]


def _is_job_related(text: str) -> bool:
    """Return True if the query is plausibly about hiring / assessments."""
    low = text.lower()
    return any(kw in low for kw in _JOB_KEYWORDS)


def recommend_assessments(query: str) -> list[dict]:
    """
    Retrieve and deduplicate the top-K SHL assessments for *query*.
    Filters out rows where name is 'Not Available'.
    """
    raw = _retrieve(query, n_results=MAX_RESULTS * 2)  # fetch extra to allow filtering

    seen: set[str] = set()
    unique: list[dict] = []
    for item in raw:
        url  = item.get("url", "")
        name = item.get("name", "Not Available")
        # Skip items where name is genuinely missing
        if url and url not in seen and name not in ("Not Available", "", "nan"):
            seen.add(url)
            unique.append(item)

    return unique[:MAX_RESULTS]


def _to_assessment_item(raw: dict) -> AssessmentItem:
    """Map a raw metadata dict to the public AssessmentItem schema."""
    return AssessmentItem(
        assessment_name=raw.get("name", "Unknown"),
        url=raw.get("url", ""),
        test_type=raw.get("test_type", "Not classified"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Application
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description=(
        "Semantic search-powered API that recommends SHL talent assessments "
        "based on a free-text job role or skill requirement."
    ),
    version="1.0.0",
    contact={"name": "SHL Scout"},
    license_info={"name": "MIT"},
)

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────


@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    response_description="Returns healthy when the API is up.",
)
def health_check() -> HealthResponse:
    """Return a simple liveness signal."""
    return HealthResponse(status="healthy")


@app.post(
    "/recommend",
    tags=["Recommendations"],
    summary="Get SHL assessment recommendations",
    response_description="Ranked list of matching SHL assessments.",
    status_code=status.HTTP_200_OK,
)
def recommend(
    body: Annotated[
        RecommendRequest,
        Body(description="Query describing the role or skills required."),
    ],
) -> RecommendResponse:
    """
    Given a free-text description of a job or required skills, return up to
    **10 deduplicated SHL assessments** ranked by semantic relevance.

    - **query**: natural-language description (e.g. *"Python developer with SQL"*)
    """
    log.info("POST /recommend  query='%s'", body.query[:120])

    # ── Off-topic check ──────────────────────────────────────────────────────
    if not _is_job_related(body.query):
        log.info("POST /recommend  → off-topic query, returning friendly message")
        return RecommendResponse(
            query=body.query,
            total_recommendations=0,
            recommendations=[],
            message=(
                "Hey there! 👋 I'm SHL Scout — I specialise in recommending "
                "talent assessments for job roles and skills. "
                "Try something like: 'Looking to hire a Python developer' or "
                "'Need assessments for a sales manager role'. What role are you hiring for?"
            ),
        )

    try:
        raw_results = recommend_assessments(body.query)
    except Exception as exc:
        log.exception("Recommendation engine error for query '%s'", body.query)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation engine failed: {exc}",
        ) from exc

    recommendations = [_to_assessment_item(r) for r in raw_results]

    # ── Friendly message when results are empty ────────────────────────────────
    msg = ""
    if not recommendations:
        msg = (
            "I couldn't find a close match in the SHL catalog for that query. "
            "Try rephrasing with a specific job title, technology, or skill — "
            "for example: 'Java developer', 'data entry clerk', or 'sales manager'."
        )

    log.info(
        "POST /recommend  → %d results for query='%s'",
        len(recommendations),
        body.query[:80],
    )

    return RecommendResponse(
        query=body.query,
        total_recommendations=len(recommendations),
        recommendations=recommendations,
        message=msg,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Local dev entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
