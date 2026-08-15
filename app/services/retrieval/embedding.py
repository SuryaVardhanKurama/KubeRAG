import time
import socket
import threading
import requests
import logfire
from app.config import settings


# ── Rate limiting (Gemini free tier) ───────────────────────────────────────────
# The free Gemini tier throttles at ~15 requests/min. Enforce a minimum interval
# between API calls and back off much longer on 429s than a plain retry loop.
_MIN_INTERVAL = 7.0  # seconds between requests (~8.5 req/min)
_RETRY_WAITS = [15, 30, 60, 120]

_last_call = [0.0]
_throttle_lock = threading.Lock()


def _throttle():
    """Sleep so that consecutive API calls stay at least _MIN_INTERVAL apart."""
    with _throttle_lock:
        wait = _MIN_INTERVAL - (time.monotonic() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()


# ── IPv4 forcing ────────────────────────────────────────────────────────────────
# This machine's IPv6 path hangs: DNS returns AAAA records first, and Python's
# HTTP stack tries them before falling back (curl uses happy-eyeballs, so it
# works). Force IPv4 resolution process-wide so all HTTPS calls (Gemini REST,
# Qdrant, Portkey) connect quickly.
_orig_getaddrinfo = socket.getaddrinfo

def _ipv4_getaddrinfo(*args, **kwargs):
    return [
        info for info in _orig_getaddrinfo(*args, **kwargs)
        if info[0] == socket.AF_INET
    ]

socket.getaddrinfo = _ipv4_getaddrinfo

BATCH_SIZE = 50
_GEMINI_DIM = 3072
_FALLBACK_DIM = 768  # all-mpnet-base-v2

_active_model = None
_model_type: str | None = None  # "gemini" or "fallback"

GEMINI_EMBED_MODEL = "gemini-embedding-2-preview"
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


# ── Gemini REST helpers ────────────────────────────────────────────────────────
# Note: uses the REST API directly instead of the langchain-google-genai SDK,
# which hangs on embed calls in this environment. Same model, same 3072-dim.

def _gemini_url(op: str) -> str:
    return f"{_GEMINI_BASE_URL}/models/{GEMINI_EMBED_MODEL}:{op}?key={settings.GEMINI_API_KEY}"


def _rest_embed(texts: list[str]) -> list[list[float]]:
    """Embed texts via Gemini batchEmbedContents (up to 100 requests per call)."""
    _throttle()
    url = _gemini_url("batchEmbedContents")
    body = {
        "requests": [
            {
                "model": f"models/{GEMINI_EMBED_MODEL}",
                "content": {"parts": [{"text": t}]},
            }
            for t in texts
        ]
    }
    resp = requests.post(url, json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return [emb["values"] for emb in data["embeddings"]]


# ── Model initialisation ───────────────────────────────────────────────────────

def _probe_gemini():
    """Try one embed call to verify Gemini is reachable. Returns True/False."""
    try:
        _rest_embed(["probe"])
        logfire.info("Gemini embeddings ready (gemini-embedding-2-preview, 3072-dim).")
        return True
    except Exception as e:
        logfire.warning(f"Gemini probe failed: {e}. Will use sentence-transformers fallback.")
        return False


def _load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading sentence-transformers fallback (all-mpnet-base-v2, 768-dim).")
    return SentenceTransformer("all-mpnet-base-v2")


def _init():
    """Initialise embedding model once per process. Called lazily on first use."""
    global _active_model, _model_type
    if _active_model is not None:
        return

    if _probe_gemini():
        _active_model = "gemini"
        _model_type = "gemini"
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Return the vector dimension for the active model. Call after _init()."""
    _init()
    return _GEMINI_DIM if _model_type == "gemini" else _FALLBACK_DIM


# ── Batch embedding with retry ─────────────────────────────────────────────────

def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type == "gemini":
        # Long backoff on 429: 15s → 30s → 60s → 120s (5 attempts total)
        for attempt in range(len(_RETRY_WAITS) + 1):
            try:
                return _rest_embed(batch)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(x in err for x in ("429", "rate", "quota", "resource_exhausted"))
                if is_rate_limit and attempt < len(_RETRY_WAITS):
                    wait = _RETRY_WAITS[attempt]
                    logfire.warning(
                        f"Gemini rate limit hit — retrying in {wait}s "
                        f"(attempt {attempt + 1}/{len(_RETRY_WAITS) + 1})."
                    )
                    time.sleep(wait)
                else:
                    logfire.error(f"Gemini embedding failed: {e}")
                    raise
        raise RuntimeError("Gemini rate limit persisted after all retries.")
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()


# ── Public API ─────────────────────────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    _init()
    if _model_type == "gemini":
        return _rest_embed([query])[0]
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings
