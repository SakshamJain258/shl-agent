# SHL Assessment Recommender Agent

A production-quality FastAPI service that acts as a conversational SHL assessment recommender. It uses semantic search over the SHL product catalog and Google Gemini to help hiring managers find the right assessments for their roles.

## Architecture

```
Client ──POST /chat──▶ FastAPI ──▶ Agent
                                    │
                          ┌─────────┴─────────┐
                          ▼                   ▼
                   FAISS Search          Gemini LLM
                   (retrieval.py)        (agent.py)
                          │                   │
                          ▼                   │
                   Catalog Items ─────────────┘
                   (catalog.json)         ▼
                                    ChatResponse
```

### Components

| File | Responsibility |
|------|---------------|
| `main.py` | FastAPI app, endpoints (`/health`, `/chat`), lifespan init |
| `agent.py` | Gemini prompt engineering, response parsing, recommendation validation |
| `retrieval.py` | Catalog download/cache, embedding with `all-MiniLM-L6-v2`, FAISS index |
| `models.py` | Pydantic schemas: `ChatRequest`, `ChatResponse`, `Recommendation` |

### Key Design Decisions

- **Stateless**: No server-side session storage. Every request carries the full conversation history.
- **Semantic retrieval**: Uses `sentence-transformers/all-MiniLM-L6-v2` embeddings + FAISS `IndexFlatIP` (cosine similarity via normalized vectors) to retrieve the top-15 most relevant catalog items per request.
- **Catalog filtering**: Pre-packaged Job Solutions are excluded; only Individual Test Solutions are indexed.
- **Hallucination guard**: Every recommended URL is validated against the real catalog before being returned.
- **Structured output**: Gemini is called with `response_mime_type="application/json"` for reliable JSON parsing.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the root of the project directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```


On first startup the service will:
1. Download the SHL catalog JSON (~2s)
2. Load the sentence-transformer model (~5s on first run, cached after)
3. Build the FAISS index (~1s)

Subsequent startups skip the download and use the cached `catalog.json`.

## API Reference

### `GET /health`

Health / liveness check.

**Response** `200 OK`:
```json
{"status": "ok"}
```

### `POST /chat`

Process a conversational turn.

**Request body** (`ChatRequest`):
```json
{
  "messages": [
    {"role": "user", "content": "I need assessments for a senior Java developer"}
  ]
}
```

**Response body** (`ChatResponse`):
```json
{
  "reply": "I'd recommend these assessments for a senior Java developer role...",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `reply` | `string` | Agent's conversational text |
| `recommendations` | `Recommendation[]` | Empty `[]` when clarifying/refusing; 1-10 items when recommending |
| `end_of_conversation` | `boolean` | `true` only when user confirms final shortlist or says goodbye |

#### Recommendation object

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | Exact assessment name from catalog |
| `url` | `string` | Exact product URL from catalog |
| `test_type` | `string` | Key codes: `A` (Ability), `B` (Biodata/SJT), `C` (Competencies), `D` (Development), `K` (Knowledge), `P` (Personality), `S` (Simulations) |

## Conversation Flow

The agent follows a structured conversation flow:

1. **Clarify** — If the request is vague, asks ONE focused question (recommendations: `[]`)
2. **Recommend** — Once role + level + purpose are clear, returns 1-10 assessments
3. **Refine** — User can add/remove constraints; agent updates the shortlist incrementally
4. **Compare** — User can ask to compare assessments (recommendations: `[]` during comparison)
5. **Confirm** — User confirms → `end_of_conversation: true` with final list

### Guardrails

- Refuses legal, compliance, or off-topic questions
- Handles missing catalog items honestly (suggests alternatives)
- Turn limit awareness: commits to best-effort after ~6-7 turns
- Validates all URLs against catalog to prevent hallucination

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |

## Example cURL

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I need to hire a mid-level Python developer for our backend team. Looking for selection assessments."}
    ]
  }'
```
