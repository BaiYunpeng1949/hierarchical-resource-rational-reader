
from __future__ import annotations

import itertools
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Tuple

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


# --------------------------- GPT interface ---------------------------

class PropositionParser:
    """Wrapper around your LLM agent; falls back to heuristic extractor.
       Adds a deterministic cache: the same sentence string -> same proposition set.
    """

    def __init__(self, llm_agent=None, role: str = "", use_llm: bool = True,
                 logger: Optional[logging.Logger] = None):
        self.llm = llm_agent
        self.role = role
        self.use_llm = (llm_agent is not None) and use_llm
        self.logger = logger
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
        if key in self._cache:
            self.cache_hits += 1
            self.last_method = "cache"
            if self.logger and self.logger.level <= logging.DEBUG:
                self.logger.debug(f"  CACHE hit for sentence: {sent[:80]}")
            return [Proposition(pred, args, sent_id, (0, len(sent))) for (pred, args) in self._cache[key]]

        self.cache_misses += 1
        self.last_method = "heuristic"
        props: Optional[List[Proposition]] = None
        if self.use_llm:
            props = self._parse_with_llm(sent, sent_id)
            if props:
                self.last_method = "llm"
        if not props:
            props = _heuristic_propositions(sent, sent_id)

        self._cache[key] = [(p.predicate, p.args) for p in props]
        if self.logger and self.logger.level <= logging.DEBUG:
            self.logger.debug(f"  CACHE store ({len(props)} props) for sentence: {sent[:80]}")

        return [Proposition(p.predicate, p.args, sent_id, (0, len(sent))) for p in props]

    def _parse_with_llm(self, sent: str, sent_id: int) -> Optional[List[Proposition]]:
        try:
            prompt = (
                "Parse this sentence into micro-structural propositions per Kintsch.\n"
                "Format strictly as A(B, C) or nested A(B, C(D)). Only propositions, comma-separated; no extra text.\n"
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
                    p_store: float = 0.35) -> List[CycleOutput]:

    if schema is None:
        schema = Schema()
    if ltm_store is None:
        ltm_store = LTMStore(p_store=p_store)

    visit_counter: Dict[int, int] = {}
    out: List[CycleOutput] = []
    total_steps = len(ep["text_reading_logs"])
    prev_wm: List[str] = []

    t_ep0 = time.perf_counter()
    for k, log in enumerate(ep["text_reading_logs"], start=1):
        t0 = time.perf_counter()

        sent_id = log["current_sentence_index"]
        visit_counter[sent_id] = visit_counter.get(sent_id, 0) + 1

        sent = _clean_join(log["sampled_words_in_sentence"])
        if not sent:
            continue

        # 1) parse micro-props
        props = parser.parse(sent, sent_id=sent_id)

        # 2) re-read gain
        strength = apply_visit_gain(visit_counter[sent_id])

        # 3) schema update + 4) macroselection (WM/gist of this cycle)
        schema.update_from_props(props, strength=strength)
        wm_props, keep_scores = macroselect_gist(props, schema, buffer_size=wm_buffer)

        # WM dynamics vs previous cycle
        wm_now = [p.signature() for p in wm_props]
        set_prev, set_now = set(prev_wm), set(wm_now)
        wm_retained = sorted(set_prev & set_now)
        wm_added = sorted(list(set_now - set_prev))
        wm_dropped = sorted(list(set_prev - set_now))

        # 5) local CI integration (optional coherence boost)
        net = build_network(props)
        integrate(net)

        # 6) LTM update (per-prop)
        ltm_updates = ltm_store.update(selected=wm_now, step=k, strength=strength, relevance_map=keep_scores)

        # record
        out.append(CycleOutput(
            step=log["step"],
            sent_id=sent_id,
            sentence_text=sent,
            propositions=[p.signature() for p in props],
            wm_kept=wm_now,
            wm_retained=wm_retained,
            wm_added=wm_added,
            wm_dropped=wm_dropped,
            ltm_updates=ltm_updates,
            schema_snapshot=dict(sorted(schema.weights.items(), key=lambda x: -x[1])[:32])
        ))

        # logs
        if logger and (k % log_every == 0 or k == total_steps):
            dt = time.perf_counter() - t0
            top_now = ltm_store.topk(k=3)
            top_keys = [key for key, _ in top_now]
            logger.info(
                f"{episode_tag} step {k}/{total_steps} | sent={sent_id} visit={visit_counter[sent_id]} | "
                f"{len(props)} props ({parser.last_method}), WM={len(wm_now)} [ret {len(wm_retained)}, +{len(wm_added)}, -{len(wm_dropped)}] | "
                f"LTM+={len(ltm_updates)} (top: {', '.join(top_keys) if top_keys else '-'}) | "
                f"{dt:.2f}s"
            )
            if logger.level <= logging.DEBUG:
                if wm_now:
                    logger.debug("  WM now: " + " | ".join(wm_now[:5]))
                if wm_retained or wm_added or wm_dropped:
                    logger.debug(f"  WM retained: {', '.join(wm_retained[:5])}")
                    logger.debug(f"  WM added:    {', '.join(wm_added[:5])}")
                    logger.debug(f"  WM dropped:  {', '.join(wm_dropped[:5])}")
                if ltm_updates:
                    lines = []
                    for u in ltm_updates[:5]:
                        lines.append(f"{u['proposition']} (visits={u['visits']}, est_recall~{u['est_recall']:.2f}, rel~{u['last_relevance']:.3f})")
                    logger.debug("  LTM updates: " + " | ".join(lines))

        prev_wm = wm_now  # carry WM to next cycle

    if logger:
        logger.info(f"{episode_tag} DONE in {time.perf_counter() - t_ep0:.2f}s | LTM size={len(ltm_store.db)}")
    return out


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
                 p_store: float = 0.35) -> Dict:
    """Run CI+Schema over episodes with rich logging, WM/LTM tracking."""
    logger = setup_logger(verbose)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # slice episodes
    data_slice = data[start:start + limit_episodes] if limit_episodes is not None else data[start:]
    logger.info(f"Loaded {len(data)} episodes; processing {len(data_slice)} (start={start}).")

    parser = PropositionParser(llm_agent=llm_agent, role=llm_role, use_llm=use_llm, logger=logger)

    results: Dict[str, List[Dict]] = {}
    for idx, ep in enumerate(data_slice, start=1):
        key = f"episode_{ep['episode_index']}_stim_{ep['stimulus_index']}_{ep['time_condition']}"
        tag = f"[{idx}/{len(data_slice)}] {key}"
        logger.info(f"{tag} BEGIN")

        schema = Schema()
        ltm_store = LTMStore(p_store=p_store)

        cycles = process_episode(ep, parser, schema=schema, wm_buffer=wm_buffer,
                                 logger=logger, episode_tag=tag, log_every=log_every,
                                 ltm_store=ltm_store, p_store=p_store)

        # per-cycle details (backward compatible shape + new fields)
        results[key] = [asdict(c) for c in cycles]
        # include episode-level LTM summary
        results[key + "__LTM"] = {"p_store": p_store, "items": ltm_store.as_dict()}
        logger.info(f"{tag} CACHE stats: hits={parser.cache_hits}, misses={parser.cache_misses}, size={len(parser._cache)}")

    return results
