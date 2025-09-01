
from __future__ import annotations

import itertools
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np


# --------------------------- Logging ---------------------------

def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("ci_pipeline")
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


# --------------------------- Data structures ---------------------------

@dataclass(frozen=True)
class Proposition:
    """A micro-proposition: predicate(arg1, arg2, ...)."""
    predicate: str
    args: Tuple[str, ...]
    sent_id: int
    span: Tuple[int, int]  # (start_char, end_char)

    def signature(self) -> str:
        return f"{self.predicate}({', '.join(self.args)})"


@dataclass
class Node:
    key: str
    prop: Proposition
    activation: float = 0.0


@dataclass
class CINetwork:
    nodes: List[Node] = field(default_factory=list)
    edges: List[Tuple[int, int, float]] = field(default_factory=list)
    index: Dict[str, int] = field(default_factory=dict)

    def add_node(self, node: Node) -> int:
        if node.key in self.index:
            return self.index[node.key]
        i = len(self.nodes)
        self.nodes.append(node)
        self.index[node.key] = i
        return i

    def add_edge(self, i: int, j: int, w: float) -> None:
        if i != j and w != 0.0:
            self.edges.append((i, j, w))


# --------------------------- Utils ---------------------------

def _clean_join(words: List[str]) -> str:
    toks = [w for w in words if w and isinstance(w, str)]
    s = " ".join(toks)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _heuristic_propositions(sent: str, sent_id: int) -> List[Proposition]:
    """Very light SVO-ish extractor as fallback when LLM isn't used/available."""
    toks = re.findall(r"[A-Za-z0-9%'-]+|[.,!?;:]", sent)
    if not toks:
        return [Proposition("mention", (sent.lower(),), sent_id, (0, len(sent)))]
    props: List[Proposition] = []
    verbs = [
        i for i, t in enumerate(toks)
        if re.match(r".*(ed|ing|s)$", t.lower()) or t.lower() in
        {"is", "are", "was", "were", "be", "am", "been", "has", "have", "had", "will"}
    ]
    for i in verbs:
        v = toks[i].lower()
        subj = next((toks[j].lower() for j in range(i - 1, -1, -1) if toks[j].isalnum()), None)
        obj = next((toks[j].lower() for j in range(i + 1, len(toks)) if toks[j].isalnum()), None)
        if subj and obj:
            props.append(Proposition(v, (subj, obj), sent_id, (0, len(sent))))
    if not props:
        head = next((t.lower() for t in toks if t.isalnum()), "mention")
        props.append(Proposition("mention", (head,), sent_id, (0, len(sent))))
    # dedupe
    out, seen = [], set()
    for p in props:
        sig = (p.predicate, p.args, p.sent_id)
        if sig not in seen:
            out.append(p)
            seen.add(sig)
    return out


# --------------------------- LLM Schema Grouper ---------------------------

@dataclass
class Cluster:
    name: str
    exemplars: List[str] = field(default_factory=list)
    member_counts: Dict[str, int] = field(default_factory=dict)

class LLMSchemaGrouper:
    def __init__(self, llm_agent, role: str, logger: Optional[logging.Logger] = None, max_report_clusters: int = 24):
        self.llm = llm_agent
        self.role = role
        self.logger = logger
        self.max_report_clusters = max_report_clusters
        self.clusters: Dict[str, Cluster] = {}

    def _brief(self) -> List[Dict[str, str]]:
        items = []
        for name, cl in self.clusters.items():
            if cl.member_counts:
                rep = max(cl.member_counts, key=cl.member_counts.get)
            elif cl.exemplars:
                rep = cl.exemplars[0]
            else:
                rep = name
            items.append({"name": name, "rep": rep})
        return items[: self.max_report_clusters]

    def assign(self, sentence: str, facets: List[str]) -> Dict[str, List[str]]:
        """
        Ask LLM to (a) assign each facet to an existing/new bucket name,
        (b) return a canonical phrasing for that facet.
        """
        try:
            brief = self._brief()
            prompt = (
                "You organize short facets into stable schema buckets while reading.\n"
                "Existing buckets (name -> representative):\n"
                f"{json.dumps(brief, ensure_ascii=False)}\n\n"
                "Assign EACH of the following facets to a bucket. Use an EXISTING bucket name if appropriate; "
                "otherwise CREATE a concise new bucket name (2–5 words). Also provide a CANONICAL phrasing for the facet "
                "(normalize near-duplicates into one phrasing; keep entities/numbers).\n\n"
                "Return ONLY JSON as a list of objects with keys: facet, bucket, canonical.\n"
                f"Facets: {json.dumps(facets, ensure_ascii=False)}"
            )
            assignments = self.llm.get_schema_assignments(self.role, prompt)  # expects a list[dict]
        except Exception as e:
            if self.logger:
                self.logger.debug(f"LLM schema assignment failed, fallback. Reason: {e}")
            assignments = [{"facet": f, "bucket": "Misc", "canonical": f} for f in facets]

        per_bucket: Dict[str, List[str]] = {}
        for obj in assignments:
            b = (obj.get("bucket") or "Misc").strip()
            can = (obj.get("canonical") or obj.get("facet") or "").strip()
            if not can:
                continue
            cl = self.clusters.setdefault(b, Cluster(name=b))
            cl.member_counts[can] = cl.member_counts.get(can, 0) + 1
            if can not in cl.exemplars:
                cl.exemplars.append(can)
            per_bucket.setdefault(b, []).append(can)
        return per_bucket

    def sentence_reps(self, per_bucket: Dict[str, List[str]], limit: int) -> List[str]:
        """Pick one representative per activated bucket, rank by GLOBAL counts, cap to limit."""
        reps = []
        for b, items in per_bucket.items():
            # pick the locally most frequent canonical
            local_counts = {}
            for it in items:
                local_counts[it] = local_counts.get(it, 0) + 1
            local_rep = max(local_counts, key=local_counts.get)
            reps.append(local_rep)

        def global_count(s: str) -> int:
            return max((cl.member_counts.get(s, 0) for cl in self.clusters.values()), default=0)

        reps = sorted(set(reps), key=global_count, reverse=True)
        return reps[:limit]

    # def export(self) -> Dict[str, Any]:
    #     out = []
    #     for name, cl in self.clusters.items():
    #         rep = max(cl.member_counts, key=cl.member_counts.get) if cl.member_counts else (cl.exemplars[0] if cl.exemplars else name)
    #         out.append({
    #             "name": name,
    #             "representative": rep,
    #             "counts": dict(sorted(cl.member_counts.items(), key=lambda kv: -kv[1]))[:50],
    #             "exemplars": cl.exemplars[:10]
    #         })
    #     return {"clusters": out}
    def export(self) -> Dict[str, Any]:
        out = []
        for name, cl in self.clusters.items():
            rep = (max(cl.member_counts, key=cl.member_counts.get)
                if cl.member_counts
                else (cl.exemplars[0] if cl.exemplars else name))
            # slice the sorted list, then convert to dict
            counts_sorted = sorted(cl.member_counts.items(), key=lambda kv: -kv[1])[:50]
            out.append({
                "name": name,
                "representative": rep,
                "counts": dict(counts_sorted),      # <-- safe
                "exemplars": cl.exemplars[:10]
            })
        return {"clusters": out}



# --------------------------- GPT interface ---------------------------

class PropositionParser:
    """Wrapper around your LLM agent; falls back to heuristic extractor.
       Adds a deterministic cache: the same sentence string -> same proposition set.
    """

    def __init__(self, llm_agent=None, role: str = "", use_llm: bool = True,
                 logger: Optional[logging.Logger] = None,
                 allow_reparse: bool = False):
        self.llm = llm_agent
        self.role = role
        self.use_llm = (llm_agent is not None) and use_llm
        self.logger = logger
        self.allow_reparse = bool(allow_reparse)
        self.last_method = "heuristic"
        # sentence -> List[Tuple[predicate, Tuple[args...]]]
        self._cache: Dict[str, List[Tuple[str, Tuple[str, ...]]]] = {}
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    @staticmethod
    def _sent_key(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip()).lower()

    def parse(self, sent: str, sent_id: int) -> List[Proposition]:
        key = self._sent_key(sent)
        # If re-parsing is NOT allowed and we have a cache entry, return it.
        if (not self.allow_reparse) and (key in self._cache):
            self.cache_hits += 1
            self.last_method = "cache"
            if self.logger and self.logger.level <= logging.DEBUG:
                self.logger.debug(f"  CACHE hit for sentence: {sent[:80]}")
            return [Proposition(pred, args, sent_id, (0, len(sent))) for (pred, args) in self._cache[key]]

        # Either cache miss or forced fresh parse
        if key in self._cache:
            # We deliberately bypass the cache but still track as a 'reparse'
            self.last_method = "reparse"
        else:
            self.cache_misses += 1
            self.last_method = "heuristic"

        props: Optional[List[Proposition]] = None
        if self.use_llm:
            props = self._parse_with_llm(sent, sent_id)
            if props:
                self.last_method = "llm" if self.last_method != "reparse" else "reparse-llm"
        if not props:
            props = _heuristic_propositions(sent, sent_id)
            if self.last_method.startswith("reparse"):
                self.last_method = "reparse-heuristic"
            else:
                self.last_method = "heuristic"

        # Update/overwrite cache with latest parse (useful for auditing)
        self._cache[key] = [(p.predicate, p.args) for p in props]
        if self.logger and self.logger.level <= logging.DEBUG:
            self.logger.debug(f"  CACHE store ({len(props)} props) for sentence: {sent[:80]}")

        return [Proposition(p.predicate, p.args, sent_id, (0, len(sent))) for p in props]

    def _parse_with_llm(self, sent: str, sent_id: int) -> Optional[List[Proposition]]:
        try:
            # prompt = (
            #     "Parse this sentence into micro-structural propositions per Kintsch.\n"
            #     "Format strictly as A(B, C) or nested A(B, C(D)). Only propositions, comma-separated; no extra text.\n"
            #     f"Sentence: {sent}"
            # )
            prompt = (
                "Parse this sentence into micro-structural propositions (Kintsch-style).\n"
                "STRICT OUTPUT: comma-separated propositions only; NO extra text.\n"
                "Each proposition must be of the form A(B, C) or nested A(B, C(D)).\n\n"
                "Coverage requirements — be EXHAUSTIVE but avoid duplicates:\n"
                "- Actions/events: use predicates like do(agent, action(object)), event(subject, object)\n"
                "- Attributives/modifiers: has_attr(entity, attribute)\n"
                "- Numbers/measurements: quantity(entity, value unit)\n"
                "- Time/temporal: time_at(event_or_state, time_expr)\n"
                "- Location: location(entity_or_event, place)\n"
                "- Causal/conditional: cause(x, y), condition(x, y)\n"
                "- Purpose/goal: purpose(x, y)\n"
                "- Membership/part-whole: part_of(x, y)\n"
                "- Coreference: coref(mention, canonical_entity)\n"
                "- Negation: negate(proposition_signature, reason)\n\n"
                "Prefer canonical nouns/verbs; keep arguments short and consistent.\n"
                "Aim for 8-15 propositions if the sentence is information rich.\n"
                f"Sentence: {sent}"
            )

            groups = self.llm.get_micro_structural_propositions(role=self.role, prompt=prompt)

            props: List[Proposition] = []
            for g in groups:
                for p in g:
                    m = re.match(r"\s*([A-Za-z0-9_%-]+)\s*\((.*)\)\s*$", p)
                    if not m:
                        continue
                    pred = m.group(1).lower()
                    args = tuple(a.strip().lower() for a in re.split(r"\s*,\s*", m.group(2).strip()) if a.strip())
                    if pred:
                        props.append(Proposition(pred, args, sent_id, (0, len(sent))))

            uniq, seen = [], set()
            for p in props:
                sig = (p.predicate, p.args, p.sent_id)
                if sig not in seen:
                    uniq.append(p); seen.add(sig)
            return uniq
        except Exception as e:
            if self.logger:
                self.logger.debug(f"LLM parse failed; fallback to heuristic. Reason: {e}")
            return None
    
    def parse_facets(self, sent: str, sent_id: int) -> List[str]:
        # LLM facets; fall back to the whole sentence if anything fails
        try:
            # prompt = FACET_PROMPT_TEMPLATE.format(SENTENCE=sent)
            prompt = (
                "You are a careful reader building sentence-level memory.\n"
                "Task: Convert the input sentence into concise facet summaries that together preserve all meaning.\n"
                "**Requirements**:\n"
                " - Write 3-8 facets (more if the sentence is information-dense).\n"
                " - One facet per line, plain text. No commas, bullets, or numbering.\n"
                " - Keep named entities, numbers, dates, units exactly as given.\n"
                " - Prefer base-form verbs (announce, expand, include).\n"
                " - Capture: actor, action, object/topic, purpose/reason, time, place, quantities, conditions/contrast, outcome, negation/modality if present.\n"
                " - Use short, declarative fragments (≈ 5-12 words).\n"
                " - Do not invent information or resolve ambiguities; use “unspecified” when missing.\n"
                " - Output only the facets; no explanations.\n"
                f"**Sentence**: {sent}"
            )
            lines = self.llm.get_facet_summaries(self.role, prompt)
            # de-dup but keep order
            out, seen = [], set()
            for s in lines:
                if s not in seen:
                    out.append(s); seen.add(s)
            return out or [sent]
        except Exception:
            return [sent] 

# --------------------------- CI integration ---------------------------

def build_network(props: List[Proposition],
                  coarg_w: float = 0.7, copred_w: float = 0.3,
                  cross_sent_decay: float = 0.85) -> CINetwork:
    net = CINetwork()
    for p in props:
        net.add_node(Node(key=p.signature(), prop=p))

    for i, j in itertools.combinations(range(len(net.nodes)), 2):
        p1, p2 = net.nodes[i].prop, net.nodes[j].prop
        dist = abs(p1.sent_id - p2.sent_id)
        dist_factor = cross_sent_decay ** dist
        arg_overlap = len(set(p1.args) & set(p2.args))
        w1 = coarg_w * arg_overlap * dist_factor
        w2 = copred_w * (1.0 if p1.predicate == p2.predicate else 0.0) * dist_factor
        w = w1 + w2
        if w > 0.0:
            net.add_edge(i, j, w)
    return net


def integrate(net: CINetwork, iters: int = 12, leak: float = 0.10, beta: float = 1.0) -> None:
    n = len(net.nodes)
    if n == 0:
        return
    W = np.zeros((n, n), dtype=float)
    for i, j, w in net.edges:
        W[i, j] += w
        W[j, i] += w
    sids = np.array([nd.prop.sent_id for nd in net.nodes], dtype=float)
    base = 1.0 + (sids.max() - sids) / max(1.0, sids.max()) * 0.2
    a = base / base.sum()
    for _ in range(iters):
        a = (1.0 - leak) * a + beta * (W @ a)
        a = np.maximum(a, 0.0)
        s = a.sum()
        if s > 0.0:
            a = a / s
    for i, val in enumerate(a):
        net.nodes[i].activation = float(val)


# --------------------------- Sentence-Ordered Memory (SOM) ---------------------------

@dataclass
class GistBoard:
    per_sent: Dict[int, List[str]] = field(default_factory=dict)
    order: List[int] = field(default_factory=list)
    per_sent_limit: int = 12
    update_policy: str = "replace"   # "replace" or "merge_topk"

    def update(self, sent_id: int, props: List[str], scores: Optional[Dict[str, float]] = None):
        # de-dup while preserving order
        seen = set(); uniq = []
        for p in props:
            if p not in seen:
                uniq.append(p); seen.add(p)

        # rank (if scores provided)
        if scores:
            uniq = sorted(uniq, key=lambda x: scores.get(x, 0.0), reverse=True)

        if self.update_policy == "replace":
            selected = uniq[: self.per_sent_limit]
        else:  # merge_topk
            prev = self.per_sent.get(sent_id, [])
            merged = prev + [p for p in uniq if p not in prev]
            if scores:
                merged = sorted(merged, key=lambda x: scores.get(x, 0.0), reverse=True)
            selected = merged[: self.per_sent_limit]

        if sent_id not in self.order:
            self.order.append(sent_id)
        self.per_sent[sent_id] = selected

    def flatten(self) -> List[str]:
        # return all propositions in sentence order
        out = []
        for sid in sorted(self.order):
            out.extend(self.per_sent.get(sid, []))
        return out


# --------------------------- Dynamic schema + macrorules ---------------------------

@dataclass
class Schema:
    weights: Dict[str, float] = field(default_factory=dict)
    decay: float = 0.90
    learn_rate: float = 0.25

    def update_from_props(self, props: List[Proposition], strength: float = 1.0) -> None:
        # decay old weights
        for k in list(self.weights.keys()):
            self.weights[k] *= self.decay
            if self.weights[k] < 1e-5:
                del self.weights[k]
        # learn from this cycle's propositions (predicate + arguments)
        seen: Dict[str, int] = {}
        for p in props:
            seen[p.predicate] = seen.get(p.predicate, 0) + 1
            for a in p.args:
                seen[a] = seen.get(a, 0) + 1
        for k, cnt in seen.items():
            self.weights[k] = self.weights.get(k, 0.0) + self.learn_rate * strength * cnt

    def relevance(self, prop: Proposition) -> float:
        return self.weights.get(prop.predicate, 0.0) + sum(self.weights.get(a, 0.0) for a in prop.args)


def macroselect_gist(props: List[Proposition], schema: Schema,
                     buffer_size: int = 5, rel_thresh: float = 0.10) -> Tuple[List[Proposition], Dict[str, float]]:
    """Return (WM list, relevance_map). Enforces argument-overlap preference and WM budget."""
    if not props:
        return [], {}
    scored = [(p, schema.relevance(p)) for p in props]
    scored.sort(key=lambda x: x[1], reverse=True)

    keep: List[Proposition] = []
    argset = set()
    for p, score in scored:
        if len(keep) < buffer_size or score >= rel_thresh:
            share = (len(argset & set(p.args)) > 0) or (not keep)
            if share or len(keep) < buffer_size // 2:
                keep.append(p)
                argset |= set(p.args)

    # collapse duplicates
    uniq, seen = [], set()
    for p in keep:
        sig = (p.predicate, p.args)
        if sig not in seen:
            uniq.append(p); seen.add(sig)

    keep_scores = {p.signature(): schema.relevance(p) for p in uniq}
    return uniq, keep_scores


# --------------------------- Re-read gain ---------------------------

def apply_visit_gain(visit_count: int, base_gain: float = 0.05) -> float:
    """Strength multiplier for schema learning when sentence is re-read."""
    return 1.0 + base_gain * max(0, visit_count - 1)


# --------------------------- LTM store ---------------------------

@dataclass
class LTMEntry:
    visits: int = 0                # number of cycles this prop was selected into WM
    first_step: Optional[int] = None
    last_step: Optional[int] = None
    total_strength: float = 0.0    # cumulative strength (with re-read gain)
    last_relevance: float = 0.0    # schema relevance at last selection
    est_recall: float = 0.0        # 1 - (1 - p)^visits


class LTMStore:
    def __init__(self, p_store: float = 0.35):
        self.db: Dict[str, LTMEntry] = {}
        self.p_store = float(p_store)

    def update(self, selected: List[str], step: int,
               strength: float, relevance_map: Dict[str, float]) -> List[Dict[str, float]]:
        """Update LTM for propositions selected this cycle. Return per-prop update dicts (for logging)."""
        updates: List[Dict[str, float]] = []
        for key in selected:
            e = self.db.get(key, LTMEntry())
            e.visits += 1
            e.first_step = e.first_step if e.first_step is not None else step
            e.last_step = step
            e.total_strength += strength
            e.last_relevance = float(relevance_map.get(key, 0.0))
            e.est_recall = 1.0 - (1.0 - self.p_store) ** e.visits
            self.db[key] = e
            updates.append({
                "proposition": key,
                "visits": e.visits,
                "est_recall": e.est_recall,
                "last_relevance": e.last_relevance
            })
        return updates

    def topk(self, k: int = 10, by: str = "est_recall") -> List[Tuple[str, LTMEntry]]:
        items = list(self.db.items())
        if by == "strength":
            items.sort(key=lambda kv: kv[1].total_strength, reverse=True)
        elif by == "visits":
            items.sort(key=lambda kv: kv[1].visits, reverse=True)
        else:
            items.sort(key=lambda kv: kv[1].est_recall, reverse=True)
        return items[:k]

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        return {k: asdict(v) for k, v in self.db.items()}


# --------------------------- Per-episode processing ---------------------------

@dataclass
class CycleOutput:
    step: int
    sent_id: int
    sentence_text: str
    propositions: List[str]
    wm_kept: List[str]
    wm_retained: List[str]
    wm_added: List[str]
    wm_dropped: List[str]
    ltm_updates: List[Dict[str, float]]
    schema_snapshot: Dict[str, float]


def process_episode(ep: Dict,
                    parser: PropositionParser,
                    schema: Optional[Schema] = None,
                    wm_buffer: int = 5,
                    logger: Optional[logging.Logger] = None,
                    episode_tag: str = "",
                    log_every: int = 1,
                    ltm_store: Optional[LTMStore] = None,
                    p_store: float = 0.35,
                    mode: str = "ci_schema",
                    som_limit: int = 12,
                    som_policy: str = "replace",
                    parse_mode: str = "llm",
                    schema_grouper: Optional[LLMSchemaGrouper] = None,
                    ) -> Tuple[List[CycleOutput], GistBoard]:

    if schema is None:
        schema = Schema()
    if ltm_store is None:
        ltm_store = LTMStore(p_store=p_store)

    visit_counter: Dict[int, int] = {}
    out: List[CycleOutput] = []
    total_steps = len(ep["text_reading_logs"])
    prev_wm: List[str] = []

    t_ep0 = time.perf_counter()

    gist_board = GistBoard(per_sent_limit=som_limit, update_policy=som_policy)

    # micro_prop_mode = parse_mode in ("llm")
    micro_prop_mode = parse_mode in ("llm", "heuristic")

    for k, log in enumerate(ep["text_reading_logs"], start=1):
        t0 = time.perf_counter()

        sent_id = log["actual_reading_sentence_index"]
        visit_counter[sent_id] = visit_counter.get(sent_id, 0) + 1

        sent = _clean_join(log["sampled_words_in_sentence"])
        if not sent:
            continue
        
        # if parse_mode == "facets":
        #     strength = apply_visit_gain(visit_counter[sent_id])
        #     wm_now = parser.parse_facets(sent, sent_id)              # list[str]
        #     keep_scores = {}                                         # no schema scores
        #     props = []                                               # avoid UnboundLocalError
        #     wm_props = []
        #     gist_board.update(sent_id, wm_now, keep_scores)          # all facets go to this sentence slot
        # elif parse_mode == "raw":
        #     # No micro-propositions; keep the exact sentence text as the single “item”
        #     props: List[Proposition] = []
        #     wm_props: List[Proposition] = []
        #     keep_scores: Dict[str, float] = {}
        #     wm_now = [sent]                       # one item per sentence
        #     # Strength still depends on revisits
        #     strength = apply_visit_gain(visit_counter[sent_id])
        #     # Update SOM directly with the raw sentence
        #     gist_board.update(sent_id, wm_now, keep_scores)
        
        # else:
        #     # 1) parse micro-props
        #     props = parser.parse(sent, sent_id=sent_id)

        #     # 2) re-read gain
        #     strength = apply_visit_gain(visit_counter[sent_id])

        #     # 3) schema update + 4) macroselection (WM/gist of this cycle)
        #     if mode == "none":
        #         # --- BYPASS: no schema learning, no macroselection, no CI coherence ---
        #         wm_props = props[:]  # keep EVERYTHING in WM/gist for this cycle
        #         keep_scores = {p.signature(): 0.0 for p in wm_props}  # default zeros
        #         # store all propositions for this sentence (ordered), no scores
        #         wm_now = [p.signature() for p in wm_props]
        #         gist_board.update(sent_id, wm_now, keep_scores)
        #     else:
        #         # --- Original CI+Schema path ---
        #         schema.update_from_props(props, strength=strength)
        #         wm_props, keep_scores = macroselect_gist(props, schema, buffer_size=wm_buffer)
        #         # store the selected WM items with scores
        #         wm_now = [p.signature() for p in wm_props] 
        #         gist_board.update(sent_id, wm_now, keep_scores)

        #     # WM dynamics vs previous cycle
        #     wm_now = [p.signature() for p in wm_props]
        
        # # ------------------ end parsing modes -------------------
        # set_prev, set_now = set(prev_wm), set(wm_now)
        # wm_retained = sorted(set_prev & set_now)
        # wm_added = sorted(list(set_now - set_prev))
        # wm_dropped = sorted(list(set_prev - set_now))

        # # 5) local CI integration (optional coherence boost)
        # # net = build_network(props)
        # # integrate(net)
        # if mode == "ci_schema" and micro_prop_mode:
        #     net = build_network(props)
        #     integrate(net)

        # # 6) LTM update (per-prop)
        # ltm_updates = ltm_store.update(selected=wm_now, step=k, strength=strength, relevance_map=keep_scores)

        # # record
        # out.append(CycleOutput(
        #     step=log["step"],
        #     sent_id=sent_id,
        #     sentence_text=sent,
        #     propositions=( [p.signature() for p in props] if micro_prop_mode else wm_now ),
        #     wm_kept=wm_now,
        #     wm_retained=wm_retained,
        #     wm_added=wm_added,
        #     wm_dropped=wm_dropped,
        #     ltm_updates=ltm_updates,
        #     schema_snapshot=(dict(sorted(schema.weights.items(), key=lambda x: -x[1])[:32]) if parse_mode != "raw" else {})
        # ))

        # # logs
        # if logger and (k % log_every == 0 or k == total_steps):
        #     dt = time.perf_counter() - t0
        #     top_now = ltm_store.topk(k=3)
        #     top_keys = [key for key, _ in top_now]
        #     items_count = len(wm_now) if not micro_prop_mode else len(wm_props)
        #     method_tag = parser.last_method if micro_prop_mode else ("facets" if parse_mode=="facets" else "raw")
        #     logger.info(
        #         f"{episode_tag} step {k}/{total_steps} | sent={sent_id} visit={visit_counter[sent_id]} | "
        #         f"{items_count} items ({method_tag}), WM={len(wm_now)} [ret {len(wm_retained)}, +{len(wm_added)}, -{len(wm_dropped)}] | "
        #         f"LTM+={len(ltm_updates)} (top: {', '.join(top_keys) if top_keys else '-'}) | "
        #         f"{dt:.2f}s"
        #     )
        #     if logger.level <= logging.DEBUG:
        #         if wm_now:
        #             logger.debug("  WM now: " + " | ".join(wm_now[:5]))
        #         if wm_retained or wm_added or wm_dropped:
        #             logger.debug(f"  WM retained: {', '.join(wm_retained[:5])}")
        #             logger.debug(f"  WM added:    {', '.join(wm_added[:5])}")
        #             logger.debug(f"  WM dropped:  {', '.join(wm_dropped[:5])}")
        #         if ltm_updates:
        #             lines = []
        #             for u in ltm_updates[:5]:
        #                 lines.append(f"{u['proposition']} (visits={u['visits']}, est_recall~{u['est_recall']:.2f}, rel~{u['last_relevance']:.3f})")
        #             logger.debug("  LTM updates: " + " | ".join(lines))

        # prev_wm = wm_now  # carry WM to next cycle

        if mode == "llm_schema":
            # 1) Get facets (parse_mode facets preferred)
            if parse_mode == "facets":
                facets = parser.parse_facets(sent, sent_id)
            elif parse_mode == "raw":
                facets = [sent]
            else:
                # if you still want to support micro-props here, you could map them to short phrases
                facets = [p.signature() for p in parser.parse(sent, sent_id=sent_id)]

            # 2) Assign to schema buckets and canonicalize
            per_bucket = schema_grouper.assign(sent, facets)

            # 3) Pick one representative per activated bucket for this sentence's SOM slot
            wm_now = schema_grouper.sentence_reps(per_bucket, limit=som_limit)
            keep_scores = {}  # counts are kept internally; we don't need scores here
            gist_board.update(sent_id, wm_now, keep_scores)

            # 4) LTM update with canonical reps (counts accumulate inside the grouper across the episode)
            ltm_updates = ltm_store.update(selected=wm_now, step=k, strength=apply_visit_gain(visit_counter[sent_id]), relevance_map=keep_scores)

            # 5) Record cycle
            out.append(CycleOutput(
                step=log["step"], sent_id=sent_id, sentence_text=sent,
                propositions=wm_now, wm_kept=wm_now,
                wm_retained=sorted(set(prev_wm) & set(wm_now)),
                wm_added=sorted(list(set(wm_now) - set(prev_wm))),
                wm_dropped=sorted(list(set(prev_wm) - set(wm_now))),
                ltm_updates=ltm_updates, schema_snapshot={}
            ))
            prev_wm = wm_now
            continue  # go to next step

    if logger:
        logger.info(f"{episode_tag} DONE in {time.perf_counter() - t_ep0:.2f}s | LTM size={len(ltm_store.db)}")
    
    return out, gist_board


# --------------------------- Entry ---------------------------

def run_pipeline(json_path: str,
                 llm_agent=None,
                 llm_role: str = "You are a careful proposition parser per Kintsch.",
                 use_llm: bool = True,
                 wm_buffer: int = 5,
                 limit_episodes: Optional[int] = None,
                 start: int = 0,
                 log_every: int = 1,
                 verbose: str = "INFO",
                 p_store: float = 0.35,
                 allow_reparse: bool = False,
                 mode: str = "ci_schema",
                 som_limit: int = 12,
                 som_policy: str = "replace",
                 parse_mode: str = "llm",
                 ) -> Dict:
    """Run CI+Schema over episodes with rich logging, WM/LTM tracking.

    Args:
        json_path: Path to episodes JSON.
        llm_agent: Optional LLM agent used for proposition parsing.
        llm_role: System prompt for LLM.
        use_llm: If False, fall back to heuristic parser.
        wm_buffer: Working-memory buffer size.
        limit_episodes: Optional limit on number of episodes.
        start: Start index in episodes.
        log_every: Log frequency (in steps).
        verbose: Logging level.
        p_store: Base probability of storing a proposition to LTM.
        allow_reparse: If True, bypass the sentence cache and re-parse the same
            sentence on each visit (useful for testing whether multiple parses
            yield additional details). If False (default), the parser caches
            and reuses propositions for identical sentence strings.
    """
    logger = setup_logger(verbose)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # slice episodes
    data_slice = data[start:start + limit_episodes] if limit_episodes is not None else data[start:]
    logger.info(f"Loaded {len(data)} episodes; processing {len(data_slice)} (start={start}).")

    # Build parser. Only use LLM when parse_mode == "llm"
    effective_use_llm = (parse_mode == "llm")
    parser = PropositionParser(llm_agent=llm_agent, role=llm_role, use_llm=effective_use_llm, logger=logger, allow_reparse=allow_reparse)

    results: Dict[str, List[Dict]] = {}
    for idx, ep in enumerate(data_slice, start=1):
        key = f"episode_{ep['episode_index']}_stim_{ep['stimulus_index']}_{ep['time_condition']}"
        tag = f"[{idx}/{len(data_slice)}] {key}"
        logger.info(f"{tag} BEGIN")

        schema = Schema()
        ltm_store = LTMStore(p_store=p_store)

        grouper = LLMSchemaGrouper(llm_agent=llm_agent, role=llm_role, logger=logger)
        # cycles, gist_board = process_episode(ep, parser, schema=schema, wm_buffer=wm_buffer, logger=logger, episode_tag=tag, log_every=log_every, 
        #     ltm_store=ltm_store, p_store=p_store, mode=mode, som_limit=som_limit, som_policy=som_policy, parse_mode=parse_mode)
        cycles, gist_board = process_episode(ep, parser, schema=schema, wm_buffer=wm_buffer, logger=logger, episode_tag=tag, log_every=log_every,
            ltm_store=ltm_store, p_store=p_store, mode=mode, som_limit=som_limit, som_policy=som_policy, parse_mode=parse_mode, schema_grouper=grouper)

        # per-cycle details (backward compatible shape + new fields)
        results[key] = [asdict(c) for c in cycles]
        # include episode-level LTM summary
        results[key + "__LTM"] = {"p_store": p_store, "items": ltm_store.as_dict()}
        results[key + "__SOM"] = {
            "policy": som_policy,
            "per_sentence": {str(sid): gist_board.per_sent[sid] for sid in sorted(gist_board.order)},
            "flat": gist_board.flatten()
        }
        if mode == "llm_schema":
            results[key + "__SCHEMA"] = grouper.export()
        logger.info(f"{tag} CACHE stats: hits={parser.cache_hits}, misses={parser.cache_misses}, size={len(parser._cache)}")

    return results


# TODO: issue: sometimes the agent performs much worse on 90s condition, because there are disorganized parsed propositions stacking together, ruining the logics.
    # TODO the solution would be dynamic ltm gist integration, as last year's version.
# NOTE: when using the LLM to answer questions, even when the given ltm is the same, generated comprehension tests could be different.

# TODO: and when generating the free recalls, do not generate other shit and non-sense words

# NOTE: some nice arguments are: