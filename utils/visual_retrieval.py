"""
utils/visual_retrieval.py
-------------------------
Optional visual document retrieval module for Temporal-RegionRAG (Stage 6).

Responsibilities
----------------
1. Define a clean ``VisualPageRetriever`` interface that accepts a query and a
   collection of document pages and returns ranked page IDs and relevance scores
   (``VisualPageHit``).
2. Provide an optional ``ColPaliVisualRetriever`` backend (``vidore/colpali-v1.2``)
   with late-interaction MaxSim scoring that activates only when ``colpali_engine``
   and model weights are available (or when a custom encoder is injected for testing),
   and never downloads multi-gigabyte models repeatedly.
3. Keep OCR/text-based retrieval (``OCRPageRetrieverFallback``) as the automatic,
   always-available fallback when ColPali is not installed, disabled, or fails.
4. Connect retrieved pages to existing table regions (``detect_table_regions``),
   extracted observations, normalized dates, and source bounding-box coordinates
   (``ConnectedPageEvidence``).
5. Explicitly distinguish page-level visual retrieval from cell/observation-level
   grounding: page-level retrieval identifies candidate pages only and does NOT
   automatically identify the exact table cell or measurement.
6. Cache page embeddings in memory and on disk (``.visual_cache/embeddings``)
   keyed by SHA-256 content hashes so pages are not re-encoded across queries.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from utils.extraction import Observation, detect_table_regions
from utils.retrieval import LexicalRetriever
from utils.temporal_verification import (
    TemporalVerificationEngine,
    VerificationReport,
    VerifiedEvidence,
    normalize_date,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Explicit scope & limitation notice (Requirement 5)
# ---------------------------------------------------------------------------

PAGE_LEVEL_LIMITATION_NOTICE = (
    "Page-level retrieval only: visual retrieval scores whole pages and does not "
    "automatically identify the exact table cell or clinical value. Individual "
    "table regions, observations, dates, and bounding boxes shown below come from "
    "the linked OCR / PDF extraction and temporal verification stages."
)

# Default cache directory inside the workspace (sandbox-writable, offline-safe)
DEFAULT_VISUAL_CACHE_DIR = Path(__file__).resolve().parent.parent / ".visual_cache"


# ---------------------------------------------------------------------------
# Environment & dependency probe
# ---------------------------------------------------------------------------

def probe_visual_environment() -> dict:
    """
    Inspect the current runtime environment for ColPali / VLM compatibility
    without importing or downloading heavy models.
    """
    import importlib.util
    import platform
    import shutil

    torch_version = None
    cuda_available = False
    mps_available = False
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        mps_available = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    except Exception:
        pass

    colpali_installed = importlib.util.find_spec("colpali_engine") is not None
    transformers_installed = importlib.util.find_spec("transformers") is not None
    peft_installed = importlib.util.find_spec("peft") is not None
    bitsandbytes_installed = importlib.util.find_spec("bitsandbytes") is not None

    try:
        _, _, free_bytes = shutil.disk_usage(Path(__file__).resolve().parent.parent)
        free_gib = round(free_bytes / (1024 ** 3), 1)
    except Exception:
        free_gib = None

    device = "cuda" if cuda_available else ("mps" if mps_available else "cpu")
    colpali_ready = bool(colpali_installed and transformers_installed)

    notes: List[str] = []
    if not colpali_installed:
        notes.append("colpali_engine is not installed (OCR/lexical fallback active).")
    if not transformers_installed:
        notes.append("transformers is not installed.")
    if device == "mps" and not cuda_available:
        notes.append(
            "Apple Silicon MPS detected (no CUDA); 4-bit bitsandbytes quantization is "
            "unavailable on macOS, so ColPali (~3B params, ~6-7 GB in bf16/fp16) requires "
            "significant unified memory."
        )

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": torch_version,
        "device": device,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "colpali_installed": colpali_installed,
        "transformers_installed": transformers_installed,
        "peft_installed": peft_installed,
        "bitsandbytes_installed": bitsandbytes_installed,
        "colpali_ready": colpali_ready,
        "free_disk_gib": free_gib,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

def make_page_id(doc_id: str, page_number: int) -> str:
    """Canonical page identifier ``'{doc_id}::p{page_number}'``."""
    return f"{doc_id}::p{max(int(page_number), 1)}"


def parse_page_id(page_id: str) -> Tuple[str, int]:
    """Inverse of :func:`make_page_id`."""
    if "::p" in page_id:
        doc_id, p_str = page_id.rsplit("::p", 1)
        try:
            return doc_id, int(p_str)
        except ValueError:
            return doc_id, 1
    return page_id, 1


@dataclass
class DocumentPage:
    """A single rendered document page supplied to a visual retriever."""
    doc_id: str
    page_number: int
    image: Image.Image

    @property
    def page_id(self) -> str:
        return make_page_id(self.doc_id, self.page_number)


@dataclass
class VisualPageHit:
    """
    Page-level retrieval result returned by any :class:`VisualPageRetriever`.

    Note: ``score`` is a page-level relevance score only; it does not locate a
    specific table cell or observation on the page.
    """
    page_id: str
    doc_id: str
    page_number: int
    score: float
    rank: int = 0
    backend_name: str = ""
    cached: bool = False
    granularity_note: str = PAGE_LEVEL_LIMITATION_NOTICE

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "page_id": self.page_id,
            "doc_id": self.doc_id,
            "page_number": self.page_number,
            "page_score": round(float(self.score), 4),
            "backend": self.backend_name,
            "embedding_cached": self.cached,
            "granularity": "page_level_only",
        }


@dataclass
class ConnectedPageEvidence:
    """
    Links a retrieved page to its detected table regions, extracted observations,
    normalized dates, and bounding-box coordinates on that page.
    """
    page_hit: VisualPageHit
    table_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    verified_evidence: List[VerifiedEvidence] = field(default_factory=list)
    dates_on_page: List[str] = field(default_factory=list)
    limitation_notice: str = PAGE_LEVEL_LIMITATION_NOTICE

    @property
    def page_id(self) -> str:
        return self.page_hit.page_id

    @property
    def doc_id(self) -> str:
        return self.page_hit.doc_id

    @property
    def page_number(self) -> int:
        return self.page_hit.page_number

    @property
    def page_score(self) -> float:
        return self.page_hit.score


@dataclass
class VisualRetrievalOutcome:
    """Complete result of running visual page retrieval + observation linking."""
    query: str
    page_hits: List[VisualPageHit]
    connected_pages: List[ConnectedPageEvidence]
    verification_report: Optional[VerificationReport]
    backend_used: str
    used_fallback: bool
    fallback_reason: Optional[str] = None
    limitation_notice: str = PAGE_LEVEL_LIMITATION_NOTICE


# ---------------------------------------------------------------------------
# Content-addressed embedding cache (Requirement 6)
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """
    Two-tier (in-memory + on-disk ``.npy``) cache for page embeddings keyed by
    SHA-256 of the page image bytes and the encoder identifier.
    """

    def __init__(self, cache_dir: Optional[Path] = None, persist_to_disk: bool = True):
        self.base_dir = Path(cache_dir) if cache_dir else (DEFAULT_VISUAL_CACHE_DIR / "embeddings")
        self.persist_to_disk = persist_to_disk
        self._mem: Dict[str, np.ndarray] = {}
        if self.persist_to_disk:
            try:
                self.base_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logger.warning("Could not create embedding cache dir %s: %s", self.base_dir, exc)
                self.persist_to_disk = False

    @staticmethod
    def compute_key(image: Image.Image, model_id: str) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        digest = hashlib.sha256(buf.getvalue()).hexdigest()[:24]
        safe_model = "".join(c if c.isalnum() else "_" for c in model_id)
        return f"{safe_model}__{digest}"

    def get(self, key: str) -> Optional[np.ndarray]:
        if key in self._mem:
            return self._mem[key]
        if self.persist_to_disk:
            path = self.base_dir / f"{key}.npy"
            if path.exists():
                try:
                    arr = np.load(path, allow_pickle=False)
                    self._mem[key] = arr
                    return arr
                except Exception as exc:
                    logger.warning("Failed to load cached embedding %s: %s", path, exc)
        return None

    def put(self, key: str, embedding: np.ndarray) -> None:
        arr = np.asarray(embedding, dtype=np.float32)
        self._mem[key] = arr
        if self.persist_to_disk:
            path = self.base_dir / f"{key}.npy"
            try:
                np.save(path, arr, allow_pickle=False)
            except Exception as exc:
                logger.warning("Failed to write cached embedding %s: %s", path, exc)

    def size(self) -> int:
        return len(self._mem)


# ---------------------------------------------------------------------------
# 1. Abstract Visual-Retriever Interface (Requirement 1)
# ---------------------------------------------------------------------------

class VisualPageRetriever(ABC):
    """
    Interface for page-level visual document retrievers.
    Implementations must return ranked ``VisualPageHit`` items with ``page_id``
    and ``score``.
    """

    name: str = "abstract-visual-retriever"

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this retriever can run in the current environment."""

    @abstractmethod
    def retrieve_pages(
        self,
        query: str,
        pages: Sequence[DocumentPage],
        top_k: int = 5,
    ) -> List[VisualPageHit]:
        """Rank ``pages`` for ``query`` and return up to ``top_k`` hits."""


# ---------------------------------------------------------------------------
# 2. Optional ColPali Backend (Requirements 2, 6, 7)
# ---------------------------------------------------------------------------

@dataclass
class ColPaliConfig:
    """
    Optional configuration for the ColPali visual retriever.
    By default ``local_files_only=True`` so the app never initiates an
    unexpected multi-gigabyte download during interactive use.
    """
    model_name: str = os.environ.get("COLPALI_MODEL_NAME", "vidore/colpali-v1.2")
    model_cache_dir: Path = Path(
        os.environ.get("COLPALI_CACHE_DIR", str(DEFAULT_VISUAL_CACHE_DIR / "models"))
    )
    embedding_cache_dir: Path = Path(
        os.environ.get("COLPALI_EMBED_CACHE_DIR", str(DEFAULT_VISUAL_CACHE_DIR / "embeddings"))
    )
    device: str = os.environ.get("COLPALI_DEVICE", "auto")
    local_files_only: bool = os.environ.get("COLPALI_ALLOW_DOWNLOAD", "0") != "1"
    enabled: bool = os.environ.get("ENABLE_COLPALI", "0") == "1"


def maxsim_score(query_vecs: np.ndarray, page_vecs: np.ndarray) -> float:
    """
    ColBERT / ColPali late-interaction MaxSim operator:
    For each query token vector (Q, D), take the maximum dot-product across all
    page patch vectors (P, D), then average over query tokens.
    """
    q = np.asarray(query_vecs, dtype=np.float32)
    p = np.asarray(page_vecs, dtype=np.float32)
    if q.ndim == 1:
        q = q[None, :]
    if p.ndim == 1:
        p = p[None, :]
    if q.size == 0 or p.size == 0 or q.shape[-1] != p.shape[-1]:
        return 0.0
    # L2-normalize rows for cosine MaxSim stability
    q_norm = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    p_norm = p / np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-8)
    sim_matrix = q_norm @ p_norm.T  # (Q, P)
    return float(np.mean(np.max(sim_matrix, axis=1)))


class ColPaliVisualRetriever(VisualPageRetriever):
    """
    Optional ColPali late-interaction page retriever.

    * Uses ``colpali_engine`` when installed and enabled, caching the loaded model
      as a singleton and caching per-page patch embeddings via :class:`EmbeddingCache`.
    * Accepts optional injectable ``image_encoder_fn`` and ``query_encoder_fn``
      callables so the full MaxSim retrieval + caching pipeline can be tested
      deterministically without downloading 6 GB of weights.
    """

    name = "colpali-late-interaction"
    _shared_model = None
    _shared_processor = None
    _shared_model_id: Optional[str] = None

    def __init__(
        self,
        config: Optional[ColPaliConfig] = None,
        embedding_cache: Optional[EmbeddingCache] = None,
        image_encoder_fn: Optional[Callable[[Image.Image], np.ndarray]] = None,
        query_encoder_fn: Optional[Callable[[str], np.ndarray]] = None,
    ):
        self.config = config or ColPaliConfig()
        self.cache = embedding_cache or EmbeddingCache(self.config.embedding_cache_dir)
        self._image_encoder_fn = image_encoder_fn
        self._query_encoder_fn = query_encoder_fn

    def is_available(self) -> bool:
        if self._image_encoder_fn is not None and self._query_encoder_fn is not None:
            return True
        if not self.config.enabled:
            return False
        env = probe_visual_environment()
        return bool(env["colpali_ready"])

    def _ensure_model_loaded(self) -> None:
        if self._image_encoder_fn is not None and self._query_encoder_fn is not None:
            return
        if not self.is_available():
            raise RuntimeError(
                "ColPali is not available or not enabled in this environment."
            )
        if (
            ColPaliVisualRetriever._shared_model is not None
            and ColPaliVisualRetriever._shared_model_id == self.config.model_name
        ):
            return

        import torch
        from colpali_engine.models import ColPali, ColPaliProcessor

        self.config.model_cache_dir.mkdir(parents=True, exist_ok=True)
        device = self.config.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        ColPaliVisualRetriever._shared_model = ColPali.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            cache_dir=str(self.config.model_cache_dir),
            local_files_only=self.config.local_files_only,
        ).to(device).eval()
        ColPaliVisualRetriever._shared_processor = ColPaliProcessor.from_pretrained(
            self.config.model_name,
            cache_dir=str(self.config.model_cache_dir),
            local_files_only=self.config.local_files_only,
        )
        ColPaliVisualRetriever._shared_model_id = self.config.model_name

    def _encode_image(self, image: Image.Image) -> Tuple[np.ndarray, bool]:
        key = self.cache.compute_key(image, self.config.model_name)
        cached = self.cache.get(key)
        if cached is not None:
            return cached, True

        if self._image_encoder_fn is not None:
            vec = np.asarray(self._image_encoder_fn(image), dtype=np.float32)
        else:
            self._ensure_model_loaded()
            import torch
            proc = ColPaliVisualRetriever._shared_processor
            model = ColPaliVisualRetriever._shared_model
            batch = proc.process_images([image.convert("RGB")]).to(model.device)
            with torch.no_grad():
                out = model(**batch)
            vec = out[0].detach().float().cpu().numpy()

        self.cache.put(key, vec)
        return vec, False

    def _encode_query(self, query: str) -> np.ndarray:
        if self._query_encoder_fn is not None:
            return np.asarray(self._query_encoder_fn(query), dtype=np.float32)
        self._ensure_model_loaded()
        import torch
        proc = ColPaliVisualRetriever._shared_processor
        model = ColPaliVisualRetriever._shared_model
        batch = proc.process_queries([query]).to(model.device)
        with torch.no_grad():
            out = model(**batch)
        return out[0].detach().float().cpu().numpy()

    def retrieve_pages(
        self,
        query: str,
        pages: Sequence[DocumentPage],
        top_k: int = 5,
    ) -> List[VisualPageHit]:
        if not query or not query.strip() or not pages:
            return []
        if not self.is_available():
            raise RuntimeError("ColPaliVisualRetriever is not available.")

        q_vecs = self._encode_query(query.strip())
        hits: List[VisualPageHit] = []
        for page in pages:
            p_vecs, was_cached = self._encode_image(page.image)
            score = maxsim_score(q_vecs, p_vecs)
            hits.append(
                VisualPageHit(
                    page_id=page.page_id,
                    doc_id=page.doc_id,
                    page_number=page.page_number,
                    score=score,
                    backend_name=self.name,
                    cached=was_cached,
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        trimmed = hits[: max(int(top_k), 1)]
        for idx, h in enumerate(trimmed, start=1):
            h.rank = idx
        return trimmed


# ---------------------------------------------------------------------------
# 3. OCR-Based Page Retriever Fallback (Requirement 3)
# ---------------------------------------------------------------------------

class OCRPageRetrieverFallback(VisualPageRetriever):
    """
    Always-available fallback that scores document pages by aggregating lexical
    relevance scores over the OCR / PDF observations extracted from each page.
    Conforms to the exact same :class:`VisualPageRetriever` interface.
    """

    name = "ocr-lexical-page-fallback"

    def __init__(
        self,
        observations: Optional[Sequence[Observation]] = None,
        lexical_retriever: Optional[LexicalRetriever] = None,
    ):
        self.observations: List[Observation] = list(observations or [])
        self.lexical = lexical_retriever or LexicalRetriever(min_score_threshold=0.01)

    def set_observations(self, observations: Sequence[Observation]) -> None:
        self.observations = list(observations)

    def is_available(self) -> bool:
        return True

    def retrieve_pages(
        self,
        query: str,
        pages: Sequence[DocumentPage],
        top_k: int = 5,
    ) -> List[VisualPageHit]:
        if not query or not query.strip() or not pages:
            return []

        obs_hits = self.lexical.retrieve(
            query,
            self.observations,
            top_k=max(len(self.observations), 20),
            min_score=0.01,
        )

        # Aggregate observation scores per (doc_id, page_number): max + 0.25 * sum of rest
        by_page: Dict[str, List[float]] = {}
        for r in obs_hits:
            pid = make_page_id(r.doc_id, max(r.page_number, 1))
            by_page.setdefault(pid, []).append(float(r.score))

        scored_hits: List[VisualPageHit] = []
        for page in pages:
            scores = sorted(by_page.get(page.page_id, []), reverse=True)
            if not scores:
                continue
            agg = scores[0] + 0.25 * sum(scores[1:])
            scored_hits.append(
                VisualPageHit(
                    page_id=page.page_id,
                    doc_id=page.doc_id,
                    page_number=page.page_number,
                    score=agg,
                    backend_name=self.name,
                    cached=False,
                )
            )

        scored_hits.sort(key=lambda h: h.score, reverse=True)
        trimmed = scored_hits[: max(int(top_k), 1)]
        for idx, h in enumerate(trimmed, start=1):
            h.rank = idx
        return trimmed


# ---------------------------------------------------------------------------
# 4. Page-to-Observation / Table / Date / Coordinate Connector (Requirement 4)
# ---------------------------------------------------------------------------

class VisualRetrievalEngine:
    """
    Coordinates visual page retrieval (with automatic OCR fallback) and connects
    retrieved pages to:
      * detected table regions (``detect_table_regions``),
      * extracted observations on those pages,
      * raw and normalized dates,
      * source bounding-box coordinates, and
      * downstream temporal verification (``TemporalVerificationEngine``).
    """

    def __init__(
        self,
        visual_retriever: Optional[VisualPageRetriever] = None,
        fallback_retriever: Optional[OCRPageRetrieverFallback] = None,
        verifier: Optional[TemporalVerificationEngine] = None,
        detect_tables: bool = True,
    ):
        self.visual_retriever = visual_retriever or ColPaliVisualRetriever()
        self.fallback_retriever = fallback_retriever or OCRPageRetrieverFallback()
        self.verifier = verifier or TemporalVerificationEngine()
        self.detect_tables = detect_tables
        self._table_cache: Dict[str, List[Tuple[int, int, int, int]]] = {}

    def retrieve_and_connect(
        self,
        query: str,
        pages: Sequence[DocumentPage],
        observations: Sequence[Observation],
        top_k_pages: int = 3,
        prefer_visual: bool = True,
    ) -> VisualRetrievalOutcome:
        self.fallback_retriever.set_observations(observations)

        used_fallback = False
        fallback_reason: Optional[str] = None
        page_hits: List[VisualPageHit] = []
        backend_used = self.fallback_retriever.name

        if prefer_visual and self.visual_retriever is not None:
            if self.visual_retriever.is_available():
                try:
                    page_hits = self.visual_retriever.retrieve_pages(
                        query, pages, top_k=top_k_pages
                    )
                    backend_used = self.visual_retriever.name
                except Exception as exc:
                    used_fallback = True
                    fallback_reason = f"Visual retriever '{self.visual_retriever.name}' failed: {exc}"
            else:
                used_fallback = True
                fallback_reason = (
                    f"Visual retriever '{self.visual_retriever.name}' is not enabled or its "
                    "dependencies/weights are not installed; using OCR page fallback."
                )

        if not prefer_visual or used_fallback:
            page_hits = self.fallback_retriever.retrieve_pages(
                query, pages, top_k=top_k_pages
            )
            backend_used = self.fallback_retriever.name

        # Index pages and observations by page_id
        page_map: Dict[str, DocumentPage] = {p.page_id: p for p in pages}
        obs_by_page: Dict[str, List[Observation]] = {}
        for o in observations:
            pid = make_page_id(o.doc_id, max(o.page_number, 1))
            obs_by_page.setdefault(pid, []).append(o)

        # Collect candidate observations from retrieved pages and run temporal verification
        retrieved_page_ids = {h.page_id for h in page_hits}
        candidate_obs: List[Observation] = []
        for h in page_hits:
            candidate_obs.extend(obs_by_page.get(h.page_id, []))

        report = self.verifier.verify(query, candidate_obs) if (query.strip() and candidate_obs) else None

        verified_by_page: Dict[str, List[VerifiedEvidence]] = {}
        if report is not None:
            for ev in report.chronological_evidence:
                pid = make_page_id(ev.doc_id, max(ev.page_number, 1))
                if pid in retrieved_page_ids:
                    verified_by_page.setdefault(pid, []).append(ev)

        connected_pages: List[ConnectedPageEvidence] = []
        for h in page_hits:
            page_obj = page_map.get(h.page_id)
            tables: List[Tuple[int, int, int, int]] = []
            if self.detect_tables and page_obj is not None:
                if h.page_id not in self._table_cache:
                    self._table_cache[h.page_id] = detect_table_regions(page_obj.image)
                tables = self._table_cache[h.page_id]

            p_obs = obs_by_page.get(h.page_id, [])
            dates_seen: List[str] = []
            for o in p_obs:
                if o.date_str:
                    nd = normalize_date(o.date_str)
                    label = f"{nd.iso_date} ('{o.date_str}')" if nd.iso_date and nd.iso_date != o.date_str else (nd.iso_date or o.date_str)
                    if label not in dates_seen:
                        dates_seen.append(label)

            connected_pages.append(
                ConnectedPageEvidence(
                    page_hit=h,
                    table_regions=tables,
                    observations=p_obs,
                    verified_evidence=verified_by_page.get(h.page_id, []),
                    dates_on_page=dates_seen,
                )
            )

        return VisualRetrievalOutcome(
            query=query,
            page_hits=page_hits,
            connected_pages=connected_pages,
            verification_report=report,
            backend_used=backend_used,
            used_fallback=used_fallback,
            fallback_reason=fallback_reason,
        )


def annotate_connected_page(
    page_image: Image.Image,
    connected: ConnectedPageEvidence,
) -> Image.Image:
    """
    Render a retrieved page with:
      * dashed/amber outlines for detected table regions (page-level structure), and
      * solid green/blue boxes for observation bounding boxes on that page.
    """
    img = page_image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    W, H = img.size

    # 1. Table regions in orange
    for idx, (tx, ty, tw, th) in enumerate(connected.table_regions, start=1):
        x1, y1 = max(0, tx), max(0, ty)
        x2, y2 = min(W, tx + tw), min(H, ty + th)
        if x2 > x1 and y2 > y1:
            draw.rectangle([x1, y1, x2, y2], outline=(217, 119, 6, 230), width=3)
            draw.text((x1 + 4, max(0, y1 - 14)), f"Table region #{idx}", fill=(217, 119, 6, 255))

    # 2. Verified/linked observations in green, other page observations in blue
    verified_ids = {ev.obs_id for ev in connected.verified_evidence}
    for obs in connected.observations:
        x, y, w, h = obs.bbox
        if w <= 0 or h <= 0:
            continue
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        if obs.obs_id in verified_ids:
            rgb = (22, 163, 74)
            fill = (*rgb, 55)
            width = 3
        else:
            rgb = (59, 130, 246)
            fill = (*rgb, 25)
            width = 1
        draw.rectangle([x1, y1, x2, y2], fill=fill, outline=(*rgb, 220), width=width)

    return Image.alpha_composite(img, overlay).convert("RGB")
