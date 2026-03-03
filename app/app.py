"""
SHL Assessment Recommender — Streamlit frontend
================================================
Standalone version: calls the ML/vector-search logic directly
(no FastAPI backend required). Works on Streamlit Cloud.
"""

import os
import sys
import time
from collections import Counter

import plotly.express as px
import streamlit as st

# ── Make sure the project root is on the path ─────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_CHARS   = 1000
MAX_RESULTS = 10

TYPE_COLORS = {
    "knowledge & skills":      ("#1d4ed8", "#dbeafe"),
    "personality & behavior":  ("#065f46", "#d1fae5"),
    "personality & behaviour": ("#065f46", "#d1fae5"),
    "ability & aptitude":      ("#5b21b6", "#ede9fe"),
    "simulations":             ("#92400e", "#fef3c7"),
    "competencies":            ("#854d0e", "#fefce8"),
    "development & 360":       ("#164e63", "#cffafe"),
    "assessment exercises":    ("#881337", "#ffe4e6"),
}
DEFAULT_BADGE = ("#374151", "#e5e7eb")
CHART_PALETTE = ["#3b82f6","#10b981","#f59e0b","#8b5cf6",
                 "#ef4444","#06b6d4","#84cc16","#f97316"]

_JOB_KEYWORDS = [
    "job", "role", "skill", "test", "assess", "hire", "hiring", "recruit",
    "developer", "engineer", "analyst", "manager", "sales", "data", "nurse",
    "python", "java", "sql", "leadership", "personality", "cognitive",
    "aptitude", "simulation", "programming", "software", "finance", "marketing",
    "recommend", "find", "suggest", "looking for", "need", "want", "candidate",
    "position", "competency", "experience", "graduate", "entry level", "work",
    "team", "communication", "customer", "service", "accounting", "clinical",
    "executive", "intern", "professional", "technical", "mechanical",
]


def _is_job_related(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in _JOB_KEYWORDS)


# ── Load ML engine (cached so it only loads once) ────────────────────────────
@st.cache_resource(show_spinner="Loading AI model …")
def load_engine():
    from embeddings.query_engine import _retrieve
    return _retrieve


def get_recommendations(query: str) -> dict:
    """Call the vector search directly and return a result dict."""
    if not _is_job_related(query):
        return {
            "recommendations": [],
            "message": (
                "Hey there! 👋 I'm SHL Scout — I specialise in recommending talent "
                "assessments for job roles and skills. Try something like: "
                "'Looking to hire a Python developer' or 'Need assessments for a sales "
                "manager role'. What role are you hiring for?"
            ),
        }

    retrieve = load_engine()
    raw = retrieve(query, n_results=MAX_RESULTS * 2)

    seen: set = set()
    unique: list[dict] = []
    for item in raw:
        url  = item.get("url", "")
        name = item.get("name", "Not Available")
        if url and url not in seen and name not in ("Not Available", "", "nan"):
            seen.add(url)
            unique.append(item)
    unique = unique[:MAX_RESULTS]

    recs = [
        {
            "assessment_name": r.get("name", "Unknown"),
            "url": r.get("url", ""),
            "test_type": r.get("test_type", "Not classified"),
        }
        for r in unique
    ]

    msg = ""
    if not recs:
        msg = (
            "I couldn't find a close match in the SHL catalog for that query. "
            "Try rephrasing with a specific job title, technology, or skill — "
            "for example: 'Java developer', 'data entry clerk', or 'sales manager'."
        )

    return {"recommendations": recs, "message": msg}


# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"], * { font-family: 'Inter', sans-serif !important; }

    .stApp { background: #f1f5f9 !important; }
    #MainMenu, footer, header { visibility: hidden !important; }
    .block-container { padding: 1.5rem 2rem 2rem !important; max-width: 100% !important; }
    section[data-testid="stSidebar"] { display: none !important; }

    .app-header {
        display: flex; align-items: center; gap: .75rem;
        padding-bottom: 1rem; border-bottom: 2px solid #e2e8f0; margin-bottom: 1.5rem;
    }
    .app-title { font-size: 1.3rem; font-weight: 800; color: #0f172a; }
    .app-sub   { font-size: .8rem; color: #94a3b8; margin-left: auto; }

    .left-card {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 14px; padding: 1.4rem 1.3rem;
        box-shadow: 0 1px 4px rgba(0,0,0,.06);
    }
    .lc-title { font-size: .72rem; font-weight: 700; color: #64748b;
                text-transform: uppercase; letter-spacing: .08em; margin-bottom: .5rem; }

    .stTextArea label { display: none !important; }
    .stTextArea > div, .stTextArea > div > div {
        background: transparent !important; border: none !important; box-shadow: none !important;
    }
    .stTextArea textarea {
        background: #f8fafc !important; border: 1.5px solid #cbd5e1 !important;
        border-radius: 10px !important; color: #0f172a !important;
        font-size: .9rem !important; line-height: 1.65 !important;
        padding: .75rem 1rem !important; resize: none !important;
        transition: border-color .18s !important;
    }
    .stTextArea textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,.12) !important;
        background: #fff !important;
    }
    .stTextArea textarea::placeholder { color: #94a3b8 !important; }
    .cc { font-size: .7rem; color: #94a3b8; text-align: right; margin-top:.1rem; }
    .cc.warn { color: #d97706; } .cc.over { color: #dc2626; }

    .stButton > button {
        background: #2563eb !important; color: #fff !important;
        font-weight: 600 !important; font-size: .92rem !important;
        border: none !important; border-radius: 10px !important;
        padding: .6rem 0 !important; width: 100% !important;
        transition: background .15s !important;
    }
    .stButton > button:hover { background: #1d4ed8 !important; }

    .ex-pill {
        display: block; font-size: .78rem; color: #475569;
        background: #f1f5f9; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: .35rem .7rem; margin-bottom: .35rem;
    }

    .results-empty {
        background: #fff; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 3rem 2rem; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,.06);
    }
    .empty-icon { font-size: 2.5rem; margin-bottom: .6rem; opacity: .3; }
    .empty-text { color: #94a3b8; font-size: .95rem; }

    .sumbar { display: flex; gap: .55rem; margin-bottom: 1rem; flex-wrap: wrap; }
    .sumchip {
        background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: .55rem .8rem; text-align: center; min-width: 72px;
        box-shadow: 0 1px 3px rgba(0,0,0,.05);
    }
    .sumchip-val { font-size: 1.5rem; font-weight: 800; color: #2563eb; line-height:1; }
    .sumchip-lbl { font-size: .62rem; color: #94a3b8; text-transform: uppercase;
                   letter-spacing: .07em; margin-top: .2rem; }

    .qecho {
        background: #eff6ff; border-left: 3px solid #2563eb;
        border-radius: 0 8px 8px 0; padding: .5rem .9rem;
        color: #1d4ed8; font-size: .83rem; margin-bottom: .9rem;
    }

    .acard {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: .9rem 1.1rem; margin-bottom: .55rem;
        box-shadow: 0 1px 3px rgba(0,0,0,.05);
        transition: border-color .18s, box-shadow .18s;
    }
    .acard:hover { border-color: #2563eb; box-shadow: 0 2px 10px rgba(37,99,235,.12); }
    .arank { font-size: .62rem; font-weight: 700; color: #2563eb;
             text-transform: uppercase; letter-spacing: .09em; margin-bottom: .2rem; }
    .aname { font-size: .95rem; font-weight: 700; color: #0f172a;
             margin-bottom: .38rem; line-height: 1.3; }
    .badge {
        display: inline-block; padding: .18rem .58rem; border-radius: 999px;
        font-size: .65rem; font-weight: 600; margin: 0 .18rem .18rem 0;
    }
    .alink { display: inline-flex; align-items: center; gap: .25rem;
             color: #2563eb; font-size: .78rem; font-weight: 500;
             text-decoration: none; margin-top: .35rem; }
    .alink:hover { color: #1e40af; text-decoration: underline; }
    .sec { font-size: .68rem; font-weight: 700; color: #94a3b8;
           text-transform: uppercase; letter-spacing: .09em; margin-bottom: .6rem; }
    .top-card { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 12px; padding: 1rem 1.1rem; }
    .botmsg { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 12px;
              padding: 1.1rem 1.3rem; color: #1e40af; font-size: .9rem; line-height: 1.7; }
    .a-warn { background: #fffbeb; border: 1px solid #fde68a;
              border-radius: 10px; padding: .75rem 1rem; color: #d97706; font-size:.88rem; }
    .div { height: 1px; background: #e2e8f0; margin: .9rem 0; }
    .js-plotly-plot .plotly .main-svg { background: transparent !important; }
    [data-testid="stSpinner"] p { color: #2563eb !important; }
    </style>
    """, unsafe_allow_html=True)


# ── UI helpers ────────────────────────────────────────────────────────────────
def _badge(test_type: str) -> str:
    k = test_type.lower().strip()
    for name, (fg, bg) in TYPE_COLORS.items():
        if name in k:
            return f'<span class="badge" style="background:{bg};color:{fg};">{name.title()}</span>'
    fg, bg = DEFAULT_BADGE
    return f'<span class="badge" style="background:{bg};color:{fg};">{test_type or "Other"}</span>'


def badges_html(tt: str) -> str:
    parts = [t.strip() for t in tt.split("|") if t.strip()]
    return "".join(_badge(p) for p in parts) if parts else _badge("Other")


def render_card(rank: int, rec: dict):
    st.markdown(f"""
    <div class="acard">
        <div class="arank">#{rank} · Match</div>
        <div class="aname">{rec.get("assessment_name", "—")}</div>
        <div>{badges_html(rec.get("test_type", ""))}</div>
        <a class="alink" href="{rec.get("url","#")}" target="_blank">🔗 SHL Catalog →</a>
    </div>""", unsafe_allow_html=True)


def render_summary(recs: list[dict]):
    tc: Counter = Counter()
    for r in recs:
        for t in r.get("test_type","Other").split("|"):
            tc[t.strip().lower()] += 1
    know = sum(v for k, v in tc.items() if "knowledge" in k or "skill" in k)
    pers = sum(v for k, v in tc.items() if "personality" in k or "behavior" in k)
    abil = sum(v for k, v in tc.items() if "ability" in k or "aptitude" in k)
    st.markdown(f"""
    <div class="sumbar">
        <div class="sumchip"><div class="sumchip-val">{len(recs)}</div><div class="sumchip-lbl">Total</div></div>
        <div class="sumchip"><div class="sumchip-val">{know}</div><div class="sumchip-lbl">Knowledge</div></div>
        <div class="sumchip"><div class="sumchip-val">{pers}</div><div class="sumchip-lbl">Personality</div></div>
        <div class="sumchip"><div class="sumchip-val">{abil}</div><div class="sumchip-lbl">Ability</div></div>
    </div>""", unsafe_allow_html=True)


def render_charts(recs: list[dict]):
    all_types: list[str] = []
    for r in recs:
        for t in r.get("test_type", "Other").split("|"):
            all_types.append(t.strip() or "Other")
    counter = Counter(all_types)
    labels, values = list(counter.keys()), list(counter.values())
    colors = CHART_PALETTE[:len(labels)]
    base = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#64748b", size=10),
                margin=dict(t=4, b=4, l=4, r=4))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="sec">Type Distribution</p>', unsafe_allow_html=True)
        fig = px.pie(names=labels, values=values, color_discrete_sequence=colors, hole=0.42)
        fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=9)
        fig.update_layout(**base, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<p class="sec">Count by Type</p>', unsafe_allow_html=True)
        fig2 = px.bar(x=labels, y=values, color=labels, color_discrete_sequence=colors)
        fig2.update_layout(**base, showlegend=False, bargap=0.3,
            xaxis=dict(showgrid=False, tickfont=dict(size=8)),
            yaxis=dict(showgrid=True, gridcolor="#e2e8f0", tickfont=dict(size=8)))
        st.plotly_chart(fig2, use_container_width=True)


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("result", None), ("elapsed", 0.0), ("last_query", ""), ("warn", "")]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    inject_css()

    st.markdown("""
    <div class="app-header">
        <span style="font-size:1.5rem;">🎯</span>
        <span class="app-title">SHL Assessment Recommender</span>
        <span class="app-sub">Semantic AI · Mistral · SHL Catalog</span>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 2], gap="large")

    # ── LEFT ──────────────────────────────────────────────────────────────────
    with left:
        st.markdown('<div class="left-card">', unsafe_allow_html=True)
        st.markdown('<p class="lc-title">🔍  Describe the role or skills</p>', unsafe_allow_html=True)

        query = st.text_area(
            label="", label_visibility="collapsed",
            placeholder="e.g. Looking to hire a Python developer with strong data analysis and SQL skills...",
            height=150, max_chars=MAX_CHARS, key="query_input",
        )

        chars = len(query)
        pct   = chars / MAX_CHARS
        cc    = "over" if pct >= 1 else ("warn" if pct >= .8 else "")
        st.markdown(f'<div class="cc {cc}">{chars}/{MAX_CHARS}</div>', unsafe_allow_html=True)

        submitted = st.button("Find Assessments →", key="submit")

        if submitted:
            q = query.strip()
            if not q:
                st.markdown('<div class="a-warn">⚠️ Please type a job description first.</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Searching SHL catalog …"):
                    t0 = time.time()
                    try:
                        data = get_recommendations(q)
                        st.session_state.result     = data
                        st.session_state.elapsed    = time.time() - t0
                        st.session_state.last_query = q
                        st.session_state.warn       = ""
                    except Exception as e:
                        st.session_state.warn = f"⚠️ Error: {e}"

        st.markdown('<div class="div"></div>', unsafe_allow_html=True)
        st.markdown('<p class="lc-title">💡 Try these examples</p>', unsafe_allow_html=True)
        for ex in [
            "Java developer with problem-solving skills",
            "Sales manager for enterprise clients",
            "Data scientist with machine learning",
            "Customer service representative",
            "Entry-level software engineer",
        ]:
            st.markdown(f'<div class="ex-pill">{ex}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── RIGHT ─────────────────────────────────────────────────────────────────
    with right:
        if st.session_state.warn:
            st.markdown(f'<div class="a-warn">{st.session_state.warn}</div>', unsafe_allow_html=True)

        elif st.session_state.result is None:
            st.markdown("""
            <div class="results-empty">
                <div class="empty-icon">🎯</div>
                <div class="empty-text">
                    Enter a job description on the left<br>and click <b>Find Assessments</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            data = st.session_state.result
            recs = data.get("recommendations", [])
            msg  = data.get("message", "")

            if not recs:
                txt = msg or "No assessments matched. Try a more specific job title or skill."
                st.markdown(
                    f'<div class="botmsg">🤖 <b>SHL Scout:</b><br><br>{txt}</div>',
                    unsafe_allow_html=True,
                )
            else:
                seen: set = set()
                unique: list[dict] = []
                for r in recs:
                    if (u := r.get("url", "")) not in seen:
                        seen.add(u); unique.append(r)
                unique = unique[:MAX_RESULTS]

                st.markdown(f'<div class="qecho">🔍 Results for: "{st.session_state.last_query}"</div>', unsafe_allow_html=True)
                render_summary(unique)

                col_cards, col_viz = st.columns([3, 2], gap="medium")
                with col_cards:
                    st.markdown('<p class="sec">Recommended Assessments</p>', unsafe_allow_html=True)
                    for i, rec in enumerate(unique, 1):
                        render_card(i, rec)
                with col_viz:
                    render_charts(unique)
                    top = unique[0]
                    st.markdown(f"""
                    <div class="top-card">
                        <div class="sec">⭐ Top Match</div>
                        <div style="font-size:.93rem;font-weight:700;color:#0f172a;margin-bottom:.4rem;">
                            {top.get("assessment_name","—")}</div>
                        <div>{badges_html(top.get("test_type",""))}</div><br>
                        <a class="alink" href="{top.get("url","#")}" target="_blank">Open on SHL Catalog →</a>
                    </div>""", unsafe_allow_html=True)

                st.markdown(
                    f'<p style="color:#cbd5e1;font-size:.68rem;margin-top:.6rem;">'
                    f'⏱ Retrieved in {st.session_state.elapsed:.2f}s</p>',
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
