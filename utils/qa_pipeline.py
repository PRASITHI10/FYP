"""
utils/qa_pipeline.py
--------------------
Evidence-grounded question answering for Temporal-RegionRAG.

Pipeline (deterministic, offline by default):

    question
      -> LexicalRetriever           (candidate evidence, utils/retrieval.py)
      -> TemporalVerificationEngine (VERIFIED / UNCERTAIN / CONFLICTING / REJECTED,
                                     utils/temporal_verification.py)
      -> EvidencePackage            (only the evidence the answer may use)
      -> AnswerGenerator            (TemplateAnswerGenerator by default; any LLM is an
                                     optional plug-in behind the same interface)
      -> QAResult                   (answer + evidence + provenance + status flags)

Design rules
------------
* Answers are produced ONLY from the supplied evidence package.
* If no evidence survives verification, the answer states that evidence is
  insufficient — nothing is inferred or invented.
* Conflicting values are reported side by side; none is selected as "correct".
* Uncertain evidence (low OCR confidence, ambiguous date, duplicate, non-standard
  unit) is reported separately and never presented as verified.
* Chronological listings describe recorded values only — no trend, causal, or
  clinical interpretation is generated.
* An optional LLM answer is accepted only if it passes a grounding check (every
  number it mentions must appear in the evidence). Otherwise — or if the model is
  unavailable or raises — the template answer is used.

This is a research prototype. It is NOT a clinical diagnostic tool.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from utils.extraction import Observation
from utils.retrieval import LexicalRetriever, RetrievalResult
from utils.temporal_verification import (
    TemporalVerificationEngine,
    VerificationReport,
    VerificationStatus,
    VerifiedEvidence,
)

RESEARCH_DISCLAIMER = (
    "Research prototype only — not a clinical diagnostic tool. Answers are assembled "
    "solely from extracted document evidence, have not been clinically validated, and "
    "must not be used for diagnosis or treatment decisions."
)

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvidencePackage:
    """The complete, closed set of evidence an answer generator may use."""
    question: str
    concept_labels: List[str]
    temporal_description: str
    is_comparison: bool
    concept_recognized: bool
    verified: List[VerifiedEvidence]
    uncertain: List[VerifiedEvidence]
    conflicting: List[VerifiedEvidence]
    rejected: List[VerifiedEvidence]
    chronological: List[VerifiedEvidence]      # verified + uncertain + conflicting, time-ordered
    citation_index: Dict[str, int]             # obs_id -> citation number [E#]

    @property
    def has_usable_evidence(self) -> bool:
        return bool(self.verified or self.uncertain or self.conflicting)

    def cite(self, ev: VerifiedEvidence) -> str:
        return f"[E{self.citation_index.get(ev.obs_id, '?')}]"


@dataclass
class GeneratedAnswer:
    text: str
    generator_name: str
    used_fallback: bool = False
    fallback_reason: Optional[str] = None


@dataclass
class QAResult:
    question: str
    retrieval_results: List[RetrievalResult]
    report: Optional[VerificationReport]
    package: EvidencePackage
    answer: GeneratedAnswer
    comparison_rows: List[dict]
    insufficient_evidence: bool
    has_conflicts: bool
    has_uncertain_ocr: bool
    has_uncertain: bool
    missing_timestamp_count: int
    disclaimer: str = RESEARCH_DISCLAIMER

    @property
    def evidence(self) -> List[VerifiedEvidence]:
        """All non-rejected evidence in chronological order (the citation order)."""
        return self.package.chronological


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_value(ev: VerifiedEvidence) -> str:
    if ev.value is None:
        return "no numeric value parsed"
    v = ev.value
    v_str = str(int(v)) if float(v).is_integer() else str(v)
    return f"{v_str} {ev.unit}".strip() if ev.unit else v_str


def _fmt_date(ev: VerifiedEvidence) -> str:
    nd = ev.normalized_date
    if nd.is_missing:
        return "undated (no timestamp in source)"
    if nd.iso_date and nd.raw_text and nd.raw_text != nd.iso_date:
        return f"{nd.iso_date} (recorded as '{nd.raw_text}')"
    return nd.iso_date or nd.raw_text or "undated"


def _fmt_source(ev: VerifiedEvidence) -> str:
    x, y, w, h = ev.bbox
    region = f"region x={x}, y={y}, w={w}, h={h}" if ev.has_valid_coordinates() else "no region coordinates"
    page = f"page {ev.page_number}" if ev.page_number > 0 else "image"
    return f"{ev.doc_id}, {page}, {region}"


def evidence_row(ev: VerifiedEvidence, citation: str) -> dict:
    """Flat, display-ready record of one evidence item with full provenance."""
    x, y, w, h = ev.bbox
    return {
        "Ref": citation,
        "Status": ev.status.value,
        "Recorded text": ev.text,
        "Value": ev.value,
        "Unit": ev.unit or "",
        "Date (normalized)": ev.normalized_date.iso_date or "",
        "Date (as recorded)": ev.raw_date_str or "",
        "Document": ev.doc_id,
        "Page": ev.page_number,
        "Region (x,y,w,h)": f"{x},{y},{w},{h}" if ev.has_valid_coordinates() else "",
        "OCR conf.": ev.observation.ocr_confidence,
        "Reason": ev.reason,
    }


def _missing_timestamp_count(rejected: Sequence[VerifiedEvidence]) -> int:
    return sum(1 for r in rejected if r.normalized_date.is_missing and "timestamp" in r.reason)


# ---------------------------------------------------------------------------
# Answer generators
# ---------------------------------------------------------------------------

class AnswerGenerator(ABC):
    """Interface for anything that turns an EvidencePackage into answer text."""

    name: str = "abstract"

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def generate(self, package: EvidencePackage) -> str:
        ...


class TemplateAnswerGenerator(AnswerGenerator):
    """
    Deterministic, evidence-only answer template. Always available.
    Every statement is a restatement of a cited evidence item.
    """

    name = "evidence-template"

    def is_available(self) -> bool:
        return True

    def generate(self, package: EvidencePackage) -> str:
        concept = ", ".join(package.concept_labels) or "the requested information"
        scope = f" {package.temporal_description}" if package.temporal_description else ""
        lines: List[str] = []

        # --- Insufficient evidence -------------------------------------------------
        if not package.has_usable_evidence:
            lines.append(
                f"**Insufficient evidence.** No observation in the supplied documents could be "
                f"verified as matching {concept}{scope}. No answer is given."
            )
            if package.rejected:
                lines.append("")
                lines.append(f"{len(package.rejected)} candidate observation(s) were examined and rejected:")
                for r in package.rejected[:5]:
                    lines.append(f"- '{r.text}' — {r.reason}")
                if len(package.rejected) > 5:
                    lines.append(f"- … and {len(package.rejected) - 5} more (see Rejected evidence).")
            return "\n".join(lines)

        if not package.concept_recognized:
            lines.append(
                "_Note: the question's concept is not in the verification lexicon, so evidence "
                "below is based on lexical matching and date checks only._"
            )
            lines.append("")

        # --- Verified evidence -----------------------------------------------------
        verified_sorted = [e for e in package.chronological if e.status == VerificationStatus.VERIFIED]
        if verified_sorted:
            distinct_dates = {e.normalized_date.iso_date for e in verified_sorted}
            if package.is_comparison or len(distinct_dates) > 1:
                lines.append(
                    f"**Verified evidence for {concept}{scope}, in chronological order** "
                    f"(values as recorded; no trend or causal interpretation is made):"
                )
            else:
                lines.append(f"**Verified evidence for {concept}{scope}:**")
            for ev in verified_sorted:
                lines.append(
                    f"- {_fmt_date(ev)}: \"{ev.text}\" — parsed value {_fmt_value(ev)} "
                    f"({_fmt_source(ev)}) {package.cite(ev)}"
                )
        else:
            lines.append(
                f"**No fully verified evidence** was found for {concept}{scope}. "
                "Only conflicting and/or uncertain evidence is available (below); it is not "
                "presented as an answer."
            )

        # --- Conflicts -------------------------------------------------------------
        if package.conflicting:
            lines.append("")
            lines.append(
                "**⚠ Conflicting evidence.** Different values were recorded for the same concept "
                "on the same date. None of them is selected as correct:"
            )
            for ev in sorted(package.conflicting, key=lambda e: e.normalized_date.sort_key):
                lines.append(
                    f"- {_fmt_date(ev)}: \"{ev.text}\" ({_fmt_source(ev)}) {package.cite(ev)}"
                )

        # --- Uncertain -------------------------------------------------------------
        if package.uncertain:
            lines.append("")
            lines.append("**⚠ Uncertain evidence** (shown for review, not used as verified):")
            for ev in sorted(package.uncertain, key=lambda e: e.normalized_date.sort_key):
                reason = ev.reason
                if ev.is_duplicate and ev.duplicate_of_id in package.citation_index:
                    reason = f"Duplicate record: same value, unit and date as [E{package.citation_index[ev.duplicate_of_id]}]"
                lines.append(
                    f"- {_fmt_date(ev)}: \"{ev.text}\" — {reason} ({_fmt_source(ev)}) {package.cite(ev)}"
                )

        # --- Excluded for missing timestamps ---------------------------------------
        n_missing = _missing_timestamp_count(package.rejected)
        if n_missing:
            lines.append("")
            lines.append(
                f"_{n_missing} otherwise-matching observation(s) were excluded because they carry "
                "no timestamp and a specific time period was requested._"
            )
        return "\n".join(lines)


def build_llm_prompt(package: EvidencePackage) -> str:
    """Strict, closed-book prompt: the model may only restate the listed evidence."""
    ev_lines = []
    for ev in package.chronological:
        ev_lines.append(
            f"{package.cite(ev)} status={ev.status.value}; text=\"{ev.text}\"; "
            f"value={_fmt_value(ev)}; date={_fmt_date(ev)}; source={_fmt_source(ev)}; "
            f"note={ev.reason}"
        )
    evidence_block = "\n".join(ev_lines) if ev_lines else "(no evidence)"
    return (
        "You are assisting with a research prototype that reads historical medical documents. "
        "You are not a clinician and must not give diagnoses, advice, trends, or causal explanations.\n"
        "Answer the question using ONLY the evidence items below. Rules:\n"
        "1. Cite every statement with its evidence reference, e.g. [E1].\n"
        "2. Do not mention any number, date, unit or fact that is not in the evidence.\n"
        "3. Report CONFLICTING items side by side without choosing one.\n"
        "4. Label UNCERTAIN items as uncertain.\n"
        "5. If the evidence does not answer the question, say the evidence is insufficient.\n\n"
        f"Question: {package.question}\n\nEvidence:\n{evidence_block}\n\nAnswer:"
    )


class CallableLLMAnswerGenerator(AnswerGenerator):
    """
    Optional adapter for any language model exposed as ``llm_fn(prompt) -> str``.

    No model, SDK, or network dependency is bundled. Pass ``llm_fn=None`` (the default)
    and the generator reports itself unavailable, so the pipeline uses the template.
    """

    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None, name: str = "optional-llm"):
        self.llm_fn = llm_fn
        self.name = name

    def is_available(self) -> bool:
        return self.llm_fn is not None

    def generate(self, package: EvidencePackage) -> str:
        if self.llm_fn is None:
            raise RuntimeError("No language model configured.")
        return str(self.llm_fn(build_llm_prompt(package)))


def _allowed_numbers(package: EvidencePackage) -> set:
    allowed: set = set()

    def add_from(text: Optional[str]) -> None:
        for n in _NUMBER_RE.findall(text or ""):
            allowed.add(float(n))

    add_from(package.question)
    for idx, ev in enumerate(package.chronological, start=1):
        allowed.add(float(idx))
        add_from(ev.text)
        add_from(ev.reason)
        add_from(ev.raw_date_str)
        add_from(ev.normalized_date.iso_date)
        add_from(ev.unit)
        add_from(ev.doc_id)
        allowed.add(float(ev.page_number))
        if ev.value is not None:
            allowed.add(float(ev.value))
        if ev.observation.ocr_confidence is not None:
            allowed.add(float(ev.observation.ocr_confidence))
        for coord in ev.bbox:
            allowed.add(float(coord))
    return allowed


def validate_grounded_answer(text: str, package: EvidencePackage) -> Tuple[bool, str]:
    """
    Reject generated text that mentions numbers absent from the evidence, or that
    makes claims without any citation while evidence exists.
    """
    if not text or not text.strip():
        return False, "empty model output"
    allowed = _allowed_numbers(package)
    # Ignore numbers inside citation markers like [E3] and ISO dates (checked via parts)
    unknown = [n for n in _NUMBER_RE.findall(text) if float(n) not in allowed]
    if unknown:
        return False, f"model output contains numbers not present in evidence: {sorted(set(unknown))}"
    if package.has_usable_evidence and not re.search(r"\[E\d+\]", text):
        return False, "model output contains no evidence citations"
    return True, "ok"


# ---------------------------------------------------------------------------
# Source-region helpers (for "open the original page")
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    VerificationStatus.VERIFIED: (22, 163, 74),
    VerificationStatus.UNCERTAIN: (217, 119, 6),
    VerificationStatus.CONFLICTING: (220, 38, 38),
    VerificationStatus.REJECTED: (120, 120, 120),
}


def get_source_page_image(
    page_images: Dict[str, List[Image.Image]],
    doc_id: str,
    page_number: int,
) -> Optional[Image.Image]:
    """Return the rendered page for (doc_id, page_number). Page 0 = standalone image."""
    pages = page_images.get(doc_id) or []
    if not pages:
        return None
    idx = 0 if page_number <= 0 else page_number - 1
    return pages[idx] if 0 <= idx < len(pages) else None


def highlight_evidence_regions(
    page_image: Image.Image,
    evidence: Sequence[VerifiedEvidence],
    labels: Optional[Sequence[str]] = None,
    line_width: int = 3,
) -> Image.Image:
    """Draw status-coloured boxes (green/amber/red) around evidence regions on a page copy."""
    base = page_image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    W, H = base.size
    for i, ev in enumerate(evidence):
        if not ev.has_valid_coordinates():
            continue
        x, y, w, h = ev.bbox
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        rgb = _STATUS_COLORS.get(ev.status, (37, 99, 235))
        draw.rectangle([x1, y1, x2, y2], fill=(*rgb, 50), outline=(*rgb, 255), width=line_width)
        label = labels[i] if labels and i < len(labels) else ev.status.value
        ty = max(0, y1 - 14)
        draw.rectangle([x1, ty, x1 + 7 * len(label) + 6, y1], fill=(*rgb, 230))
        draw.text((x1 + 3, ty + 1), label, fill=(255, 255, 255, 255))
    return Image.alpha_composite(base, overlay).convert("RGB")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _temporal_description(report: VerificationReport) -> str:
    parts = [c.raw_query_segment for c in report.query_spec.temporal_constraints
             if c.constraint_type != "unbounded" and c.raw_query_segment]
    return f"for {', '.join(parts)}" if parts else ""


class EvidenceQAPipeline:
    """Retrieve -> verify -> package -> generate (with template fallback)."""

    def __init__(
        self,
        retriever: Optional[LexicalRetriever] = None,
        verifier: Optional[TemporalVerificationEngine] = None,
        candidate_k: int = 25,
        min_score: float = 0.05,
    ):
        self.retriever = retriever or LexicalRetriever()
        self.verifier = verifier or TemporalVerificationEngine()
        self.candidate_k = candidate_k
        self.min_score = min_score
        self.template = TemplateAnswerGenerator()

    def answer(
        self,
        question: str,
        observations: List[Observation],
        generator: Optional[AnswerGenerator] = None,
    ) -> QAResult:
        question = (question or "").strip()

        # Exclude extraction-failure placeholders from the evidence pool
        pool = [o for o in observations if (o.text or "").strip()]

        retrieved = self.retriever.retrieve(
            question, pool, top_k=self.candidate_k, min_score=self.min_score
        ) if question else []
        candidates = [r.observation for r in retrieved]

        report = self.verifier.verify(question, candidates) if candidates else None

        if report is not None:
            chronological = list(report.chronological_evidence)
            package = EvidencePackage(
                question=question,
                concept_labels=report.query_spec.target_concept_labels,
                temporal_description=_temporal_description(report),
                is_comparison=report.query_spec.is_comparison,
                concept_recognized=bool(report.query_spec.target_concepts),
                verified=report.verified,
                uncertain=report.uncertain,
                conflicting=report.conflicting,
                rejected=report.rejected,
                chronological=chronological,
                citation_index={ev.obs_id: i for i, ev in enumerate(chronological, start=1)},
            )
        else:
            from utils.temporal_verification import parse_clinical_query
            spec = parse_clinical_query(question) if question else None
            package = EvidencePackage(
                question=question,
                concept_labels=spec.target_concept_labels if spec else [],
                temporal_description="",
                is_comparison=bool(spec and spec.is_comparison),
                concept_recognized=bool(spec and spec.target_concepts),
                verified=[], uncertain=[], conflicting=[], rejected=[],
                chronological=[], citation_index={},
            )

        answer = self._generate(package, generator)

        comparison_rows: List[dict] = []
        distinct_dates = {e.normalized_date.iso_date for e in package.chronological
                          if e.normalized_date.iso_date}
        if package.is_comparison or len(distinct_dates) > 1:
            comparison_rows = [
                {
                    "Ref": package.cite(ev),
                    "Date": ev.normalized_date.iso_date or "undated",
                    "Recorded text": ev.text,
                    "Value": ev.value,
                    "Unit": ev.unit or "",
                    "Status": ev.status.value,
                    "Document": ev.doc_id,
                    "Page": ev.page_number,
                }
                for ev in package.chronological
            ]

        uncertain_ocr = any(
            e.observation.ocr_uncertain
            or (e.observation.ocr_confidence is not None
                and e.observation.ocr_confidence < self.verifier.ocr_confidence_threshold)
            for e in package.uncertain
        )

        return QAResult(
            question=question,
            retrieval_results=retrieved,
            report=report,
            package=package,
            answer=answer,
            comparison_rows=comparison_rows,
            insufficient_evidence=not package.verified,
            has_conflicts=bool(package.conflicting),
            has_uncertain_ocr=uncertain_ocr,
            has_uncertain=bool(package.uncertain),
            missing_timestamp_count=_missing_timestamp_count(package.rejected),
        )

    def _generate(self, package: EvidencePackage, generator: Optional[AnswerGenerator]) -> GeneratedAnswer:
        template_text = self.template.generate(package)

        if generator is None or isinstance(generator, TemplateAnswerGenerator):
            return GeneratedAnswer(template_text, self.template.name)

        if not generator.is_available():
            return GeneratedAnswer(template_text, self.template.name, True,
                                   f"'{generator.name}' is not available")

        if not package.has_usable_evidence:
            # Never let a model answer when there is no evidence.
            return GeneratedAnswer(template_text, self.template.name, True,
                                   "insufficient evidence — model not called")
        try:
            text = generator.generate(package)
        except Exception as exc:  # model/network failure must not break the app
            return GeneratedAnswer(template_text, self.template.name, True,
                                   f"'{generator.name}' failed: {exc}")

        ok, why = validate_grounded_answer(text, package)
        if not ok:
            return GeneratedAnswer(template_text, self.template.name, True,
                                   f"'{generator.name}' output rejected: {why}")
        return GeneratedAnswer(text.strip(), generator.name)
