"""
utils/retrieval.py
------------------
Retrieval module for Temporal-RegionRAG.

Responsibilities:
1. Accept natural-language questions (e.g. "What was the blood pressure on 15 Jan 1998?",
   "Compare glucose levels in 1998 and 2002", "Show heart rate over time").
2. Retrieve relevant OCR text and structured observations using an offline,
   fast lexical-ranking baseline (BM25 Okapi + domain field-matching).
3. Rank matching observations and retain their source document ID, page number,
   date, and spatial coordinates (bounding boxes).
4. Multi-date comparison: When a query requires comparing observations across
   different dates/years, retrieve and return relevant results from each target date.
5. Highlight retrieved regions on source document images at their exact coordinates.
6. Provide clear status / empty results when no relevant evidence matches.

Architecture:
- Completely decoupled from the Streamlit UI.
- No heavy LLM or visual retrieval models required.
- Fully offline and deterministic.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image, ImageDraw, ImageFont

from utils.extraction import Observation


# ---------------------------------------------------------------------------
# Stopwords and Tokenization Helpers
# ---------------------------------------------------------------------------

DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "did", "do",
    "does", "doing", "don't", "down", "during", "each", "few", "find", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it",
    "its", "itself", "let's", "me", "more", "most", "my", "myself", "no", "nor", "not",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "please", "same", "she", "should", "show", "so", "some", "such",
    "tell", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "were", "weren't", "what", "when", "where",
    "which", "while", "who", "whom", "why", "with", "won't", "would", "you", "your",
    "yours", "yourself", "yourselves",
}

# Regex for tokenization (preserves alphanumeric words, numbers, and medical units)
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[./-][A-Za-z0-9]+)?")

# Regex to detect 4-digit years (e.g. 1998, 2024)
_YEAR_PATTERN = re.compile(r"\b(19\d\d|20\d\d)\b")

# Regex to detect common date formats in questions
_DATE_QUERY_PATTERNS = [
    re.compile(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"),
    re.compile(r"\b(\d{1,2}[./]\d{1,2}[./]\d{4})\b"),
    re.compile(
        r"\b(\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
        r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)(?:\s+\d{4})?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
        r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,?\s+\d{4})?)\b",
        re.IGNORECASE,
    ),
]

# Keywords signaling temporal comparison or progression
COMPARISON_KEYWORDS: Set[str] = {
    "compare", "comparison", "comparing", "trend", "trends", "progression",
    "difference", "differences", "change", "changes", "changed", "over time",
    "history", "timeline", "vs", "versus", "between",
}


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Split text into lowercase alphanumeric tokens.
    Numbers, years, and clinical measurements are preserved.
    """
    if not text:
        return []
    tokens = [t.lower() for t in _TOKEN_PATTERN.findall(text)]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in DEFAULT_STOPWORDS and len(t) > 1]
    return tokens


def extract_temporal_anchors(query: str) -> List[str]:
    """
    Extract date or year strings explicitly mentioned in a query.
    Examples:
        "Compare glucose in 1998 and 2002" -> ["1998", "2002"]
        "What was BP on 15 Jan 1998?" -> ["15 Jan 1998"]
    """
    anchors: List[str] = []
    # 1. Check full date patterns
    for pat in _DATE_QUERY_PATTERNS:
        matches = pat.findall(query)
        for m in matches:
            if m not in anchors:
                anchors.append(m)

    # 2. Check 4-digit years
    years = _YEAR_PATTERN.findall(query)
    for y in years:
        if not any(y in a for a in anchors):
            anchors.append(y)

    return anchors


def is_comparison_query(query: str) -> bool:
    """
    Determine if a natural language query is asking to compare across dates/time.
    Criteria:
    - Contains comparison keywords ("compare", "trend", "over time", "vs", etc.)
    - OR mentions 2 or more distinct dates/years.
    """
    q_lower = query.lower()
    for kw in COMPARISON_KEYWORDS:
        if kw in q_lower:
            return True
    anchors = extract_temporal_anchors(query)
    if len(anchors) >= 2:
        return True
    return False


# ---------------------------------------------------------------------------
# Retrieval Result Dataclass
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """
    A single ranked observation retrieved for a query, preserving all
    source document metadata, spatial coordinates, and ranking evidence.
    """
    observation: Observation
    score: float
    rank: int = 0
    matched_terms: List[str] = field(default_factory=list)
    matched_dates: List[str] = field(default_factory=list)
    temporal_anchor: Optional[str] = None

    @property
    def doc_id(self) -> str:
        return self.observation.doc_id

    @property
    def page_number(self) -> int:
        return self.observation.page_number

    @property
    def date_str(self) -> Optional[str]:
        return self.observation.date_str

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self.observation.bbox

    @property
    def text(self) -> str:
        return self.observation.text

    @property
    def value(self) -> Optional[float]:
        return self.observation.value

    @property
    def unit(self) -> Optional[str]:
        return self.observation.unit

    @property
    def source_type(self) -> str:
        return self.observation.source_type

    @property
    def ocr_confidence(self) -> Optional[float]:
        return self.observation.ocr_confidence

    @property
    def ocr_uncertain(self) -> bool:
        return self.observation.ocr_uncertain

    def has_valid_coordinates(self) -> bool:
        """Return True if observation carries non-zero bounding box coordinates."""
        _, _, w, h = self.bbox
        return w > 0 and h > 0

    def to_dict(self) -> dict:
        d = self.observation.to_dict()
        d["rank"] = self.rank
        d["retrieval_score"] = round(self.score, 4)
        d["matched_terms"] = ", ".join(self.matched_terms)
        d["matched_dates"] = ", ".join(self.matched_dates)
        d["temporal_anchor"] = self.temporal_anchor or ""
        return d


# ---------------------------------------------------------------------------
# Lexical Retriever (BM25 Okapi + Clinical Domain Matching)
# ---------------------------------------------------------------------------

class LexicalRetriever:
    """
    Fast, offline lexical ranking baseline based on BM25 Okapi with
    clinical observation field boosts (text, value, unit, date).

    Parameters
    ----------
    k1 : float
        BM25 term frequency saturation parameter (default 1.5).
    b : float
        BM25 document length normalization parameter (default 0.75).
    min_score_threshold : float
        Minimum score required for a document to be considered relevant.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        min_score_threshold: float = 0.05,
    ):
        self.k1 = k1
        self.b = b
        self.min_score_threshold = min_score_threshold

    def _observation_to_document_text(self, obs: Observation) -> str:
        """Compose a searchable document string from an observation."""
        parts = [obs.text or ""]
        if obs.unit:
            parts.append(obs.unit)
        if obs.date_str:
            parts.append(obs.date_str)
        if obs.value is not None:
            parts.append(str(obs.value))
        return " ".join(parts)

    def retrieve(
        self,
        query: str,
        observations: List[Observation],
        top_k: int = 10,
        min_score: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """
        Rank and retrieve matching observations for a natural-language question.

        Features:
        - BM25 Okapi lexical scoring over observation text, unit, date, and value.
        - Exact term and phrase matching boosts.
        - Automatic multi-date comparison handling: when the query requests
          comparing multiple dates or asks about changes over time, results
          are gathered across distinct dates to ensure balanced temporal evidence.
        - Retains source doc_id, page_number, date_str, and bbox coordinates.
        - Returns [] when no relevant observations meet the score threshold.
        """
        if not query or not query.strip() or not observations:
            return []

        threshold = min_score if min_score is not None else self.min_score_threshold
        query_text = query.strip()
        query_tokens = tokenize(query_text, remove_stopwords=True)
        if not query_tokens:
            # Fall back to un-stopped tokens if all were filtered
            query_tokens = tokenize(query_text, remove_stopwords=False)
            if not query_tokens:
                return []

        temporal_anchors = extract_temporal_anchors(query_text)
        is_comparison = is_comparison_query(query_text)

        # 1. Build observation corpus and inverted document frequency (IDF)
        corpus_tokens: List[List[str]] = [
            tokenize(self._observation_to_document_text(obs), remove_stopwords=False)
            for obs in observations
        ]
        n_docs = len(corpus_tokens)
        if n_docs == 0:
            return []

        avg_doc_len = sum(len(d) for d in corpus_tokens) / max(n_docs, 1)

        # Document frequencies for query terms
        doc_freqs: Dict[str, int] = Counter()
        for doc in corpus_tokens:
            unique_terms = set(doc)
            for t in query_tokens:
                if t in unique_terms:
                    doc_freqs[t] += 1

        # Calculate BM25 Okapi IDF for each query token
        idf: Dict[str, float] = {}
        for t in query_tokens:
            df = doc_freqs.get(t, 0)
            # Okapi IDF with floor to prevent negative weights
            idf_val = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            idf[t] = max(idf_val, 0.1)

        # 2. Score each observation
        scored_candidates: List[Tuple[float, Observation, List[str], List[str]]] = []

        q_lower = query_text.lower()

        for idx, (obs, doc_terms) in enumerate(zip(observations, corpus_tokens)):
            doc_len = len(doc_terms)
            term_counts = Counter(doc_terms)

            bm25_score = 0.0
            matched_terms: List[str] = []

            for q_term in set(query_tokens):
                tf = term_counts.get(q_term, 0)
                if tf > 0:
                    matched_terms.append(q_term)
                    # BM25 term weighting
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(avg_doc_len, 1.0)))
                    bm25_score += idf[q_term] * (numerator / denominator)

            # Exact text match bonus
            obs_text_lower = (obs.text or "").lower()
            if obs_text_lower and obs_text_lower in q_lower:
                bm25_score += 1.5

            # Temporal match checking
            matched_dates: List[str] = []
            obs_date = (obs.date_str or "").lower()
            for anchor in temporal_anchors:
                a_lower = anchor.lower()
                if a_lower in obs_date or a_lower in obs_text_lower:
                    matched_dates.append(anchor)
                    bm25_score += 1.0  # Temporal relevance bonus

            # Field match bonuses
            if obs.unit and obs.unit.lower() in q_lower:
                bm25_score += 0.5

            if bm25_score >= threshold:
                scored_candidates.append((bm25_score, obs, matched_terms, matched_dates))

        if not scored_candidates:
            return []

        # Sort all candidates descending by score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # 3. Handle Temporal Multi-Date Comparison vs. Standard Retrieval
        results: List[RetrievalResult] = []

        if is_comparison and len(scored_candidates) > 1:
            # If explicit temporal anchors exist, pick best hits per anchor
            if len(temporal_anchors) >= 2:
                selected_obs_ids: Set[str] = set()
                anchor_results: List[RetrievalResult] = []

                for anchor in temporal_anchors:
                    a_lower = anchor.lower()
                    for score, obs, m_terms, m_dates in scored_candidates:
                        if obs.obs_id in selected_obs_ids:
                            continue
                        obs_date_full = f"{obs.date_str or ''} {obs.text or ''}".lower()
                        if a_lower in obs_date_full:
                            selected_obs_ids.add(obs.obs_id)
                            anchor_results.append(
                                RetrievalResult(
                                    observation=obs,
                                    score=score,
                                    matched_terms=m_terms,
                                    matched_dates=m_dates if m_dates else [anchor],
                                    temporal_anchor=anchor,
                                )
                            )
                            break  # Top result for this anchor

                # Add other top candidates if space remains up to top_k
                for score, obs, m_terms, m_dates in scored_candidates:
                    if len(anchor_results) >= top_k:
                        break
                    if obs.obs_id not in selected_obs_ids:
                        selected_obs_ids.add(obs.obs_id)
                        anchor_results.append(
                            RetrievalResult(
                                observation=obs,
                                score=score,
                                matched_terms=m_terms,
                                matched_dates=m_dates,
                            )
                        )
                results = anchor_results

            else:
                # General comparison/trend: group candidates across distinct dates
                date_groups: Dict[str, List[Tuple[float, Observation, List[str], List[str]]]] = {}
                for item in scored_candidates:
                    date_key = item[1].date_str or f"doc_{item[1].doc_id}"
                    date_groups.setdefault(date_key, []).append(item)

                selected_obs_ids: Set[str] = set()
                # Round-robin pick from each distinct date group to guarantee diversity
                while len(results) < top_k and any(date_groups.values()):
                    for d_key in list(date_groups.keys()):
                        if not date_groups[d_key]:
                            continue
                        score, obs, m_terms, m_dates = date_groups[d_key].pop(0)
                        if obs.obs_id not in selected_obs_ids:
                            selected_obs_ids.add(obs.obs_id)
                            results.append(
                                RetrievalResult(
                                    observation=obs,
                                    score=score,
                                    matched_terms=m_terms,
                                    matched_dates=m_dates,
                                    temporal_anchor=obs.date_str,
                                )
                            )
                            if len(results) >= top_k:
                                break
        else:
            # Standard single-topic / focused retrieval
            for score, obs, m_terms, m_dates in scored_candidates[:top_k]:
                results.append(
                    RetrievalResult(
                        observation=obs,
                        score=score,
                        matched_terms=m_terms,
                        matched_dates=m_dates,
                    )
                )

        # Assign 1-indexed ranks
        for rank_idx, res in enumerate(results, start=1):
            res.rank = rank_idx

        return results


# ---------------------------------------------------------------------------
# UI Visual Region Highlighting Helpers
# ---------------------------------------------------------------------------

# Palette of distinct highlight colors for ranked results
_RANK_COLORS = [
    (220, 38, 38),   # Rank 1: Vibrant Red
    (37, 99, 235),   # Rank 2: Bright Blue
    (16, 185, 129),  # Rank 3: Emerald Green
    (217, 119, 6),   # Rank 4: Amber / Orange
    (147, 51, 234),  # Rank 5: Violet / Purple
    (13, 148, 136),  # Rank 6: Teal
]


def highlight_retrieved_regions(
    page_image: Image.Image,
    results: List[RetrievalResult],
    target_page: Optional[int] = None,
    line_width: int = 3,
    draw_rank_badges: bool = True,
) -> Image.Image:
    """
    Overlay bounding boxes and rank badges on a document page image for
    retrieved observations.

    Parameters
    ----------
    page_image : Image.Image
        The PIL image of the document page.
    results : List[RetrievalResult]
        Retrieved results to highlight.
    target_page : Optional[int]
        If specified, only observations matching this page_number are drawn.
    line_width : int
        Border thickness of bounding boxes.
    draw_rank_badges : bool
        If True, draws a `#rank` badge with background color above the box.

    Returns
    -------
    Image.Image
        Annotated PIL image with highlighted source regions.
    """
    annotated = page_image.convert("RGBA").copy()
    overlay = Image.new("RGBA", annotated.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_final = ImageDraw.Draw(annotated)

    img_w, img_h = annotated.size

    for res in results:
        # Check page match if specified
        if target_page is not None and res.page_number > 0 and res.page_number != target_page:
            continue

        if not res.has_valid_coordinates():
            continue

        x, y, w, h = res.bbox
        # Clamp to image boundaries
        x1 = max(0, min(x, img_w - 1))
        y1 = max(0, min(y, img_h - 1))
        x2 = max(0, min(x + w, img_w))
        y2 = max(0, min(y + h, img_h))

        if x2 <= x1 or y2 <= y1:
            continue

        color_rgb = _RANK_COLORS[(res.rank - 1) % len(_RANK_COLORS)]
        fill_rgba = (*color_rgb, 45)       # 18% opacity tint fill
        border_rgba = (*color_rgb, 230)    # 90% opacity border

        # Draw semi-transparent filled box
        draw_overlay.rectangle([x1, y1, x2, y2], fill=fill_rgba, outline=border_rgba, width=line_width)

        # Draw Rank Badge
        if draw_rank_badges:
            badge_text = f"#{res.rank} (score: {res.score:.2f})"
            badge_h = 16
            badge_w = len(badge_text) * 7 + 8
            badge_y1 = max(0, y1 - badge_h)
            badge_y2 = y1

            # Badge background
            draw_overlay.rectangle(
                [x1, badge_y1, x1 + badge_w, badge_y2],
                fill=(*color_rgb, 240),
            )

    # Merge overlay with image
    merged = Image.alpha_composite(annotated, overlay).convert("RGB")
    draw_merged = ImageDraw.Draw(merged)

    # Re-draw text labels sharply on merged RGB image
    if draw_rank_badges:
        for res in results:
            if target_page is not None and res.page_number > 0 and res.page_number != target_page:
                continue
            if not res.has_valid_coordinates():
                continue
            x, y, w, h = res.bbox
            x1 = max(0, min(x, img_w - 1))
            y1 = max(0, min(y, img_h - 1))
            badge_text = f"#{res.rank} (score: {res.score:.2f})"
            badge_h = 16
            badge_y1 = max(0, y1 - badge_h)
            draw_merged.text((x1 + 4, badge_y1 + 2), badge_text, fill=(255, 255, 255))

    return merged


def crop_observation_region(
    page_image: Image.Image,
    bbox: Tuple[int, int, int, int],
    padding: int = 12,
) -> Optional[Image.Image]:
    """
    Extract a cropped visual snippet of an observation's bounding box region.
    Returns None if coordinates are missing or invalid.
    """
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None

    img_w, img_h = page_image.size
    crop_x1 = max(0, x - padding)
    crop_y1 = max(0, y - padding)
    crop_x2 = min(img_w, x + w + padding)
    crop_y2 = min(img_h, y + h + padding)

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    return page_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
