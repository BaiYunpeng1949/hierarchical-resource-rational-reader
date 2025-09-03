# schema_llm.py
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set

@dataclass
class SchemaSlot:
    rep: str = ""
    members: Dict[str, int] = field(default_factory=dict)
    sent_ids: Set[int] = field(default_factory=set)

    def add(self, canonical: str, sent_id: int):
        self.members[canonical] = self.members.get(canonical, 0) + 1
        self.sent_ids.add(sent_id)
        if not self.rep or self.members[canonical] >= self.members.get(self.rep, 0):
            self.rep = canonical

class SchemaLLM:
    """
    LLM-maintained schema memory:
      - extract sentence facets (fluent mini-summaries)
      - assign/canonicalize facets under schema buckets
      - maintain dynamic main schemas (top-K by freq)
    """

    def __init__(self, llm_agent, role: str, logger: Optional[logging.Logger] = None,
                 max_report_clusters: int = 64, top_k_activation: int = 10):
        self.llm = llm_agent
        self.role = role
        self.logger = logger
        self.max_report_clusters = max_report_clusters
        self.top_k_activation = top_k_activation
        self.clusters: Dict[str, SchemaSlot] = {}
        self.schema_freq: Dict[str, int] = {}
        self.main_schemas: List[str] = []

    # ---------- Facets ----------
    def facets(self, sentence: str) -> List[str]:
        prompt = (
            "You are a careful reader building sentence-level memory.\n"
            "Task: convert the input sentence into concise facet summaries that together preserve all meaning.\n"
            "Rules:\n"
            "- Write 3-8 facets (more if information-dense).\n"
            "- One facet per line, plain text. No commas, bullets, or numbering.\n"
            "- Keep named entities, numbers, dates, units exactly as given.\n"
            "- Prefer base-form verbs (announce, expand, include).\n"
            "- Capture: actor, action, topic/object, purpose/reason, time, place, quantities, conditions/contrast, outcome, negation/modality if present.\n"
            "- Use short, declarative fragments (≈ 5-12 words).\n"
            "- Do not invent information; use 'unspecified' when missing.\n"
            f"Sentence: {sentence}"
        )
        try:
            lines = self.llm.get_facet_summaries(self.role, prompt)
            seen = set(); out = []
            for s in lines:
                s = s.strip()
                if s and s not in seen:
                    out.append(s); seen.add(s)
            return out or [sentence]
        except Exception as e:
            if self.logger: self.logger.debug(f"[SchemaLLM] facets() fail, fallback to sentence. Reason: {e}")
            return [sentence]

    # ---------- Assignment / Clustering ----------
    def assign(self, facets: List[str]) -> Dict[str, List[str]]:
        brief = []
        for name, slot in self.clusters.items():
            if len(brief) >= self.max_report_clusters: break
            rep = slot.rep or (next(iter(slot.members.keys())) if slot.members else name)
            brief.append({"name": name, "rep": rep})
        prompt = (
            "You organize short facets into stable schema buckets while reading.\n"
            "Existing buckets (name -> representative):\n"
            f"{json.dumps(brief, ensure_ascii=False)}\n\n"
            "Assign EACH facet to a bucket. Use an EXISTING bucket name when appropriate; "
            "otherwise CREATE a concise new bucket name (2-5 words). Also provide a CANONICAL phrasing "
            "for the facet (normalize near-duplicates; keep entities/numbers).\n"
            "Return ONLY JSON as a list of objects with keys: facet, bucket, canonical.\n"
            f"Facets: {json.dumps(facets, ensure_ascii=False)}"
        )
        try:
            rows = self.llm.get_schema_assignments(self.role, prompt)
        except Exception as e:
            if self.logger: self.logger.debug(f"[SchemaLLM] assign() fail, fallback Misc. Reason: {e}")
            rows = [{"facet": f, "bucket": "Misc", "canonical": f} for f in facets]

        per_bucket: Dict[str, List[str]] = {}
        for r in rows or []:
            bucket = (r.get("bucket") or "Misc").strip()
            canonical = (r.get("canonical") or r.get("facet") or "").strip()
            if canonical:
                per_bucket.setdefault(bucket, []).append(canonical)
        return per_bucket

    def update_clusters(self, per_bucket: Dict[str, List[str]], sent_id: int) -> None:
        for name, items in per_bucket.items():
            slot = self.clusters.setdefault(name, SchemaSlot())
            for c in items:
                slot.add(c, sent_id)
                self.schema_freq[name] = self.schema_freq.get(name, 0) + 1
        self.main_schemas = [name for name, _ in sorted(self.schema_freq.items(), key=lambda kv: -kv[1])][: self.top_k_activation]

    # ---------- Representatives for this sentence ----------
    def sentence_representatives(self, per_bucket: Dict[str, List[str]], per_schema_limit: int = 1, cap: int = 12) -> List[str]:
        cands: List[str] = []
        for _, items in per_bucket.items():
            local: Dict[str, int] = {}
            for it in items: local[it] = local.get(it, 0) + 1
            reps = [kv[0] for kv in sorted(local.items(), key=lambda kv: -kv[1])[:per_schema_limit]]
            cands.extend(reps)

        def global_member_count(s: str) -> int:
            m = 0
            for slot in self.clusters.values():
                m = max(m, slot.members.get(s, 0))
            return m

        uniq, seen = [], set()
        for s in sorted(set(cands), key=lambda x: global_member_count(x), reverse=True):
            if s not in seen:
                uniq.append(s); seen.add(s)
        return uniq[:cap]

    # ---------- Export ----------
    def export(self, schema_limit: int = 10, per_schema_limit: int = 4) -> Dict[str, Any]:
        ordered = sorted(self.clusters.items(), key=lambda kv: -self.schema_freq.get(kv[0], 0))
        out = []
        for name, slot in ordered[:schema_limit]:
            members_sorted = sorted(slot.members.items(), key=lambda kv: -kv[1])[:per_schema_limit]
            out.append({
                "name": name,
                "representative": slot.rep,
                "members": dict(members_sorted),
                "support_sent_ids": sorted(list(slot.sent_ids))
            })
        schema_flat: List[str] = []
        for item in out:
            if item["representative"]: schema_flat.append(item["representative"])
            schema_flat.extend(item["members"].keys())
        return {"main_schemas": self.main_schemas, "schemas": out, "schema_flat": schema_flat}
