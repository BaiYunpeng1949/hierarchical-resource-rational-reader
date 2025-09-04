
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from .llm_agent import LLMAgent

# ---------------------------
# Helpers
# ---------------------------

def canon(text: str) -> str:
    """Canonicalize for dedupe: lowercase, alnum+space, single spaces."""
    if text is None:
        return ""
    t = text.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def is_too_noisy(text: str) -> bool:
    """Reject very noisy/gibberish lines quickly."""
    if not text or len(text.strip()) < 3:
        return True
    letters = sum(ch.isalpha() for ch in text)
    ratio = letters / max(1, len(text))
    return ratio < 0.3  # many punctuation/garbage

def join_tokens(tokens: List[str]) -> str:
    """Reconstruct a sentence hypothesis from sampled tokens, skipping empties."""
    toks = [t for t in tokens or [] if t and t.strip()]
    s = " ".join(toks).strip()
    # light cleanup for stray punctuation spacing
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s

# ---------------------------
# Data structures
# ---------------------------

@dataclass
class FacetEntry:
    text: str
    count: int = 0
    evidence: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class SchemaBucket:
    name: str
    canonical: str
    facets: Dict[str, FacetEntry] = field(default_factory=dict)

    def add_facet(self, facet_text: str, ev: Dict[str, Any]):
        f_key = canon(facet_text)
        if not f_key:
            return
        if f_key not in self.facets:
            self.facets[f_key] = FacetEntry(text=facet_text, count=0, evidence=[])
        self.facets[f_key].count += 1
        self.facets[f_key].evidence.append(ev)

    def to_dict(self):
        return {
            "name": self.name,
            "canonical": self.canonical,
            "facets": [
                {"text": f.text, "count": f.count, "evidence": f.evidence}
                for f in self.facets.values()
            ],
        }

class SchemaStore:
    """Holds schemas -> facets with dedupe; supports pretty outline export."""

    def __init__(self):
        self._schemas: Dict[str, SchemaBucket] = {}  # canon_schema -> bucket

    def get_or_create(self, schema_name: str) -> SchemaBucket:
        s_key = canon(schema_name) or "misc"
        if s_key not in self._schemas:
            self._schemas[s_key] = SchemaBucket(name=schema_name or "Misc", canonical=s_key)
        return self._schemas[s_key]

    def add(self, schema_name: str, facet_text: str, evidence: Dict[str, Any]):
        bucket = self.get_or_create(schema_name)
        bucket.add_facet(facet_text, evidence)

    def main_schemas(self, top_k: int = 5) -> List[Tuple[str, int]]:
        items = []
        for b in self._schemas.values():
            total = sum(f.count for f in b.facets.values())
            items.append((b.name, total))
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:top_k]

    def to_outline(self) -> str:
        """Simple nested bullet outline."""
        lines = []
        for name, _ in self.main_schemas(top_k=9999):
            b = self._schemas[canon(name)]
            lines.append(f"- {b.name}")
            # order by count desc
            facets_sorted = sorted(b.facets.values(), key=lambda f: (-f.count, f.text))
            for f in facets_sorted:
                lines.append(f"  - {f.text} (x{f.count})")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v.to_dict() for k, v in self._schemas.items()}

# ---------------------------
# Pipeline
# ---------------------------

class LTMPipeline:
    """
    One-cycle-per-sentence: extract facets -> assign schemas -> update store
    with dedupe, and provide a robust outline.
    """

    def __init__(self, llm: LLMAgent, facet_lines_cap: int = 8):
        self.llm = llm
        self.store = SchemaStore()
        self.facet_lines_cap = facet_lines_cap

    # ---- prompts ----

    @staticmethod
    def _role_facet() -> str:
        return (
            "You are an expert reader who extracts compact facets (key ideas) "
            "from noisy sentences. Be robust to grammar/word noise."
        )

    @staticmethod
    def _prompt_facet(sentence: str, cap: int) -> str:
        return (
            "Task: extract at most {cap} compact facets (≤12 words) from the sentence.\n"
            "- Keep only meaningful content; ignore gibberish.\n"
            "- Be non-redundant; each line is ONE facet.\n"
            "- Output: one facet per line. NO numbering, NO extra text.\n\n"
            f"Sentence: {sentence}"
        ).format(cap=cap)

    @staticmethod
    def _role_assign() -> str:
        return (
            "You cluster facets into schemas (topics). "
            "Return strict JSON only."
        )

    @staticmethod
    def _prompt_assign(facets: List[str]) -> str:
        # Provide examples/formatting inside the prompt for robustness
        facet_block = "\n".join(f"- {f}" for f in facets)
        return (
            "Cluster the facets into schemas. For EACH facet, produce a JSON object with keys:\n"
            '  - "facet": original facet string\n'
            '  - "schema": short human-readable schema name\n'
            '  - "canonical": lowercase canonical schema key (a-z0-9 and spaces only)\n\n'
            "Rules:\n"
            "- Use existing schema names consistently across facets.\n"
            "- Prefer stable, general labels (e.g., 'city bike program', 'STEM education').\n"
            "- If a facet is noise/meaningless, assign schema 'misc' and canonical 'misc'.\n"
            "- Output JSON array ONLY. No commentary.\n\n"
            "Facets:\n"
            f"{facet_block}"
        )

    # ---- core ----

    def process_cycle(self, sentence: str, evidence: Dict[str, Any]):
        if is_too_noisy(sentence):
            return

        facets: List[str] = []
        try:
            facets = self.llm.get_facet_summaries(
                role=self._role_facet(),
                prompt=self._prompt_facet(sentence, self.facet_lines_cap),
            )
        except Exception as e:
            facets = []

        # fallback: use the sentence itself if LLM returns nothing
        if not facets:
            facets = [sentence]

        # Assign schemas
        try:
            assigns = self.llm.get_schema_assignments(
                role=self._role_assign(),
                prompt=self._prompt_assign(facets),
            )
        except Exception as e:
            # fallback: put everything under Misc
            assigns = [{"facet": f, "schema": "Misc", "canonical": "misc"} for f in facets]

        # Integrate deterministically with dedupe
        for a in assigns:
            facet = a.get("facet") or ""
            schema = a.get("schema") or "Misc"
            if is_too_noisy(facet):
                continue
            self.store.add(schema, facet, evidence)

    # ---- finalization ----

    def finalize(self) -> Dict[str, Any]:
        return {
            "main_schemas": self.store.main_schemas(top_k=10),
            "outline": self.store.to_outline(),
            "schemas": self.store.to_dict(),
        }

# ---------------------------
# Trial runner
# ---------------------------

def run_trial(llm: LLMAgent, trial: Dict[str, Any], max_steps: Optional[int] = None) -> Dict[str, Any]:
    """
    trial: a dict with keys episode_index, stimulus_index, time_condition, text_reading_logs
    """
    pipe = LTMPipeline(llm)
    logs = trial.get("text_reading_logs", [])

    for i, rec in enumerate(logs[: (max_steps or len(logs))]):
        sent = join_tokens(rec.get("sampled_words_in_sentence", []))
        evidence = {
            "episode_index": trial.get("episode_index"),
            "stimulus_index": trial.get("stimulus_index"),
            "time_condition": trial.get("time_condition"),
            "step": rec.get("step"),
            "sentence_index": rec.get("current_sentence_index"),
        }
        pipe.process_cycle(sent, evidence)

    gist = pipe.finalize()
    gist["episode_index"] = trial.get("episode_index")
    gist["stimulus_index"] = trial.get("stimulus_index")
    gist["time_condition"] = trial.get("time_condition")
    gist["total_time"] = trial.get("total_time")
    return gist
