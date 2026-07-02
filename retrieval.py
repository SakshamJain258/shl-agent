"""
Catalog loading, embedding, and FAISS-based semantic search for SHL assessments.

Responsibilities:
- Download the SHL product catalog JSON on first run, cache locally
- Filter to Individual Test Solutions (exclude pure "Pre-packaged Job Solutions")
- Build sentence-transformer embeddings for each catalog item
- Store embeddings in a FAISS IndexFlatIP (cosine similarity via normalized vectors)
- Expose search_catalog() for semantic retrieval of relevant assessments
"""

import json
import logging
import os
import time
from pathlib import Path

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATALOG_URL = (
    "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/"
    "shl_product_catalog.json"
)
CATALOG_FILE = Path(__file__).parent / "catalog.json"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 15
REQUEST_TIMEOUT_SECONDS = 30

# ---------------------------------------------------------------------------
# Module-level state (initialized once via init_catalog)
# ---------------------------------------------------------------------------

_catalog_items: list[dict] = []
_faiss_index: faiss.IndexFlatIP | None = None
_embed_model: SentenceTransformer | None = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _download_catalog() -> list[dict]:
    """Download the SHL product catalog JSON from the remote URL.

    Returns:
        A list of catalog item dicts.

    Raises:
        requests.RequestException: If the download fails.
    """
    logger.info("Downloading catalog from %s …", CATALOG_URL)
    response = requests.get(CATALOG_URL, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    # The SHL catalog contains invalid control characters inside JSON strings.
    # strict=False tells the parser to tolerate them instead of raising.
    data = json.loads(response.text, strict=False)
    # Save locally for subsequent runs
    with open(CATALOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Catalog saved to %s (%d items)", CATALOG_FILE, len(data))
    return data


def _load_catalog() -> list[dict]:
    """Load the catalog from local cache, downloading first if missing.

    Returns:
        A list of raw catalog item dicts.
    """
    if CATALOG_FILE.exists():
        logger.info("Loading catalog from local cache: %s", CATALOG_FILE)
        with open(CATALOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return _download_catalog()


def _is_individual_test(item: dict) -> bool:
    """Determine whether a catalog item is an Individual Test Solution.

    An item is excluded if:
    - Its 'keys' list contains ONLY "Pre-packaged Job Solutions"
    - Its name or description strongly indicates it is a job-solution bundle

    Args:
        item: A single catalog item dict.

    Returns:
        True if the item should be kept (individual test), False otherwise.
    """
    keys = item.get("keys", [])

    # Exclude if keys contain ONLY "Pre-packaged Job Solutions"
    if keys and all(k.strip() == "Pre-packaged Job Solutions" for k in keys):
        return False

    # Heuristic: exclude items whose name/description signals a job bundle
    name_lower = (item.get("name") or "").lower()
    desc_lower = (item.get("description") or "").lower()
    bundle_signals = [
        "job solution",
        "pre-packaged solution",
        "job-solution bundle",
    ]
    for signal in bundle_signals:
        if signal in name_lower or signal in desc_lower:
            # Only exclude if there are no individual-test keys alongside
            individual_keys = [
                k for k in keys if k.strip() != "Pre-packaged Job Solutions"
            ]
            if not individual_keys:
                return False

    return True


def _build_text_representation(item: dict) -> str:
    """Build a rich text string for embedding from a catalog item.

    The representation concatenates key fields so the embedding captures
    product name, measurement constructs, seniority levels, duration,
    supported languages, and the full description.

    Args:
        item: A single catalog item dict.

    Returns:
        A single string suitable for embedding.
    """
    name = item.get("name", "")
    keys = ", ".join(item.get("keys", []))
    job_levels = ", ".join(item.get("job_levels", []))
    duration = item.get("duration", "")
    languages = ", ".join(item.get("languages", []))
    description = item.get("description", "")

    return (
        f"{name}. "
        f"Keys: {keys}. "
        f"Job levels: {job_levels}. "
        f"Duration: {duration}. "
        f"Languages: {languages}. "
        f"Description: {description}"
    )


def _keys_to_abbreviation(keys: list[str]) -> str:
    """Convert a list of SHL key category names to short codes.

    Mapping (first-letter heuristic refined for SHL conventions):
        A  = Ability & Aptitude
        B  = Biodata & Situational Judgement
        C  = Competencies
        D  = Development / 360
        K  = Knowledge & Skills
        P  = Personality & Behavior
        S  = Simulations

    Unknown keys are represented by their first letter uppercased.

    Args:
        keys: List of key strings from the catalog item.

    Returns:
        Comma-separated abbreviation string (e.g. "A,K,P").
    """
    mapping: dict[str, str] = {
        "ability & aptitude": "A",
        "biodata & situational judgement": "B",
        "competencies": "C",
        "development & 360": "D",
        "knowledge & skills": "K",
        "personality & behavior": "P",
        "personality & behaviour": "P",
        "simulations": "S",
        "pre-packaged job solutions": "J",
    }
    codes: list[str] = []
    for k in keys:
        code = mapping.get(k.strip().lower(), k.strip()[0].upper() if k.strip() else "?")
        if code not in codes:
            codes.append(code)
    return ",".join(codes)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init_catalog() -> None:
    """Initialize the catalog, embeddings, and FAISS index.

    This must be called exactly once at application startup (via FastAPI lifespan).
    It downloads/loads the catalog, filters items, embeds them, and builds the
    FAISS inner-product index.
    """
    global _catalog_items, _faiss_index, _embed_model

    start = time.perf_counter()

    # 1. Load raw catalog
    raw_items = _load_catalog()
    logger.info("Raw catalog contains %d items", len(raw_items))

    # 2. Filter to individual tests
    _catalog_items = [item for item in raw_items if _is_individual_test(item)]
    logger.info(
        "Filtered to %d individual test items (excluded %d)",
        len(_catalog_items),
        len(raw_items) - len(_catalog_items),
    )

    # 3. Pre-compute abbreviations and store on each item for downstream use
    for item in _catalog_items:
        item["_test_type_abbrev"] = _keys_to_abbreviation(item.get("keys", []))

    # 4. Build text representations
    texts = [_build_text_representation(item) for item in _catalog_items]

    # 5. Load embedding model
    logger.info("Loading sentence-transformer model: %s …", EMBEDDING_MODEL_NAME)
    _embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 6. Encode all items
    logger.info("Encoding %d catalog items …", len(texts))
    embeddings = _embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype=np.float32)

    # 7. Build FAISS index (inner product on normalised vectors ≡ cosine similarity)
    dimension = embeddings.shape[1]
    _faiss_index = faiss.IndexFlatIP(dimension)
    _faiss_index.add(embeddings)
    logger.info(
        "FAISS index built: %d vectors, dim=%d (%.2fs total)",
        _faiss_index.ntotal,
        dimension,
        time.perf_counter() - start,
    )


def search_catalog(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Search the catalog for items most relevant to *query*.

    Uses the sentence-transformer model to embed the query and performs
    an approximate nearest-neighbor search against the FAISS index.

    Args:
        query: A natural-language search string derived from the conversation.
        top_k: Number of results to return (default 15).

    Returns:
        A list of up to *top_k* catalog item dicts, ordered by relevance.

    Raises:
        RuntimeError: If the catalog has not been initialized.
    """
    if _embed_model is None or _faiss_index is None:
        raise RuntimeError("Catalog not initialized. Call init_catalog() first.")

    query_vec = _embed_model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype=np.float32)

    distances, indices = _faiss_index.search(query_vec, top_k)

    results: list[dict] = []
    for idx in indices[0]:
        if 0 <= idx < len(_catalog_items):
            results.append(_catalog_items[idx])

    logger.debug("search_catalog(query=%r, top_k=%d) → %d results", query, top_k, len(results))
    return results


def get_valid_urls() -> set[str]:
    """Return the set of all URLs present in the loaded catalog.

    Used by the agent to validate that recommended URLs actually exist
    in the catalog (guard against hallucination).

    Returns:
        A set of URL strings.
    """
    return {item.get("link", "") for item in _catalog_items}
