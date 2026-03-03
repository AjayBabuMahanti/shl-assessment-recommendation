SHL Intelligent Assessment Recommendation System

AI-powered semantic recommendation system that suggests the most relevant SHL assessments based on natural language hiring queries or job descriptions.

1. Problem Statement

Recruiters often struggle to manually identify the right SHL assessments using keyword-based filtering.

This project replaces traditional filtering with:

Semantic search

LLM-based query understanding

Balanced technical + behavioral recommendations

Automated evaluation using Mean Recall@10

2. System Architecture

User Query
→ LLM Query Analysis (Gemini)
→ Query Embedding (SentenceTransformer)
→ Vector Search (ChromaDB)
→ Balanced Ranking Logic
→ Top 5–10 SHL Assessments
→ FastAPI Backend → Streamlit Frontend

3. Core Components
3.1 Data Pipeline

Scraped 500+ Individual Test Solutions

Extracted structured fields:

Assessment Name

URL

Description

Duration

Test Type

Adaptive Support

Remote Support

3.2 Embedding & Retrieval

Model: all-MiniLM-L6-v2

Vector Store: ChromaDB (persistent storage)

Retrieval: Top-20 similarity search

Post-processing: Balanced filtering by test type

3.3 LLM Query Understanding

Model: Gemini 1.5 Flash

Extracts:

Technical skills

Soft skills

Job role

Assessment requirements

Enables balanced recommendations across domains.

3.4 Evaluation

Metric Used:

Recall@10 = Relevant Assessments in Top 10 / Total Relevant Assessments
Mean Recall@10 = Average Recall@10 across all queries

Used labeled dataset for validation and iteration.

3.5 Backend API

Endpoints:

GET /health
POST /recommend

Response format:

{
"query": "...",
"total_recommendations": 5,
"recommendations": [
{
"assessment_name": "...",
"url": "...",
"test_type": "..."
}
]
}

3.6 Frontend Application

Built with Streamlit:

Modern SaaS-style UI

Structured recommendation cards

Test-type distribution visualization

Pie chart and bar chart analytics

Proper error handling

4. Project Structure

SHL-Recommendation-System/

scraper/
embeddings/
evaluation/
api/
app/
data/
vector_db/
requirements.txt
README.md

5. Setup Instructions
Clone Repository

git clone <repo_url>
cd SHL-Recommendation-System

Create Virtual Environment

python -m venv venv
venv\Scripts\activate

Install Dependencies

pip install -r requirements.txt

Add Gemini API Key

Inside recommendation_engine.py:

GEMINI_API_KEY = "your_api_key_here"

6. Running the Application
Start Backend API

uvicorn api.main:app --reload

Visit:
http://127.0.0.1:8000/docs

Start Web App

streamlit run app/app.py

Visit:
http://localhost:8501

7. Evaluation

To compute Mean Recall@10:

python evaluation/evaluate.py

Outputs:

Recall@10 per query

Final Mean Recall@10

8. Technology Stack

Scraping: Selenium
Embeddings: SentenceTransformers
Vector Store: ChromaDB
LLM: Gemini
Backend: FastAPI
Frontend: Streamlit
Evaluation: Custom Recall@10

9. Design Decisions

Semantic search instead of keyword filtering

Balanced technical + personality recommendations

LLM-driven query interpretation

Modular production-ready architecture

10. Future Improvements

Cross-encoder reranking

Hybrid BM25 + semantic search

Direct API-based scraping

Dockerized deployment

Caching layer for performance

Author

Ajay Babu Mahanti
AI & Data Systems
