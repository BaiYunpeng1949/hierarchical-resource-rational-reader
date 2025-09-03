# ci_schema_pipeline.py
from __future__ import annotations
import json, logging, os, re, tempfile, time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from llm_agent import LLMAgent  # your existing LLM client
from schema_llm import SchemaLLM

# -------- Logging --------
def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("ci_pipeline")
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
        h.setFormatter(fmt); logger.addHandler(h)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger

# -------- Utils --------
def _clean_join(words: List[str]) -> str:
    toks = [w for w in words if w and isinstance(w, str)]
    s = " ".join(toks)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _atomic_write_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(path),
                                     suffix=".tmp", encoding="utf-8") as tf:
        json.dump(obj, tf, ensure_ascii=False, indent=2)
        tmp = tf.name
    os.replace(tmp, path)

def _load_json_or_empty(path: str) -> dict:
    if not os.path.exists(path): return {}
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f) or {}
    except Exception: return {}

def apply_visit_gain(visit_count: int, base_gain: float = 0.05) -> float:
    return 1.0 + base_gain * max(0, visit_count - 1)

# -------- Minimal LTM (for continuity with old tools) --------
@dataclass
class LTMEntry:
    visits: int = 0
    first_step: Optional[int] = None
    last_step: Optional[int] = None
    total_strength: float = 0.0
    last_relevance: float = 0.0
    est_recall: float = 0.0

class LTMStore:
    def __init__(self, p_store: float = 0.35):
        self.db: Dict[str, LTMEntry] = {}
        self.p_store = float(p_store)

    def update(self, selected: List[str], step: int, strength: float, relevance_map: Dict[str, float]) -> List[Dict[str, float]]:
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
            updates.append({"proposition": key, "visits": e.visits, "est_recall": e.est_recall, "last_relevance": e.last_relevance})
        return updates

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        return {k: asdict(v) for k, v in self.db.items()}

# -------- SOM (trace-only) --------
@dataclass
class GistBoard:
    per_sent: Dict[int, List[str]] = field(default_factory=dict)
    order: List[int] = field(default_factory=list)
    per_sent_limit: int = 12
    update_policy: str = "merge_topk"

    def update(self, sent_id: int, items: List[str]):
        seen = set(); uniq = []
        for x in items:
            if x not in seen:
                uniq.append(x); seen.add(x)
        if self.update_policy == "replace":
            selected = uniq[: self.per_sent_limit]
        else:
            prev = self.per_sent.get(sent_id, [])
            merged = prev + [x for x in uniq if x not in prev]
            selected = merged[: self.per_sent_limit]
        if sent_id not in self.order: self.order.append(sent_id)
        self.per_sent[sent_id] = selected

    def flatten(self) -> List[str]:
        out: List[str] = []
        for sid in sorted(self.order): out.extend(self.per_sent.get(sid, []))
        return out

# -------- Per-step record --------
@dataclass
class CycleOutput:
    step: int
    sent_id: int
    sentence_text: str
    wm_kept: List[str]
    wm_retained: List[str]
    wm_added: List[str]
    wm_dropped: List[str]
    ltm_updates: List[Dict[str, float]]

# -------- Episode processing (LLM schema-centric) --------
def process_episode(ep: Dict[str, Any],
                    llm_agent,
                    logger: logging.Logger,
                    som_limit: int = 12,
                    som_policy: str = "merge_topk",
                    p_store: float = 0.35,
                    use_actual_index: bool = True,
                    ) -> Tuple[List[CycleOutput], Dict[str, Any], Dict[str, Any]]:

    gb = GistBoard(per_sent_limit=som_limit, update_policy=som_policy)
    ltm = LTMStore(p_store=p_store)
    schema = SchemaLLM(llm_agent=llm_agent, role="You are a schema maintainer for a reading agent.", logger=logger,
                       max_report_clusters=64, top_k_activation=10)

    visit_counter: Dict[int, int] = {}
    out: List[CycleOutput] = []
    text_logs = ep.get("text_reading_logs", [])
    total_steps = len(text_logs)
    prev_wm: List[str] = []

    for k, log in enumerate(text_logs, start=1):
        sent_id = int(log.get("actual_reading_sentence_index", log.get("current_sentence_index", -1))) if use_actual_index else int(log.get("current_sentence_index", -1))
        if sent_id < 0: continue
        sent = _clean_join(log.get("sampled_words_in_sentence", []))
        if not sent: continue

        visit_counter[sent_id] = visit_counter.get(sent_id, 0) + 1
        strength = apply_visit_gain(visit_counter[sent_id])

        # 1) facets
        facets = schema.facets(sent)
        # 2) assign to schemas + update global
        per_bucket = schema.assign(facets)
        schema.update_clusters(per_bucket, sent_id=sent_id)
        # 3) sentence representatives (trace)
        wm_now = schema.sentence_representatives(per_bucket, per_schema_limit=1, cap=som_limit)
        gb.update(sent_id, wm_now)
        # 4) LTM update
        ltm_updates = ltm.update(selected=wm_now, step=k, strength=strength, relevance_map={})

        prev_set, now_set = set(prev_wm), set(wm_now)
        wm_retained = sorted(prev_set & now_set)
        wm_added = sorted(list(now_set - prev_set))
        wm_dropped = sorted(list(prev_set - now_set))

        out.append(CycleOutput(
            step=int(log.get("step", k)),
            sent_id=sent_id,
            sentence_text=sent,
            wm_kept=wm_now,
            wm_retained=wm_retained,
            wm_added=wm_added,
            wm_dropped=wm_dropped,
            ltm_updates=ltm_updates
        ))
        prev_wm = wm_now

        if (k % max(1, (total_steps // 5 or 1))) == 0:
            logger.info(f"  step {k}/{total_steps} | sent={sent_id} | wm_now={len(wm_now)} | main_schemas={schema.main_schemas[:5]}")

    som_view = {"policy": gb.update_policy,
                "per_sentence": {str(sid): gb.per_sent[sid] for sid in sorted(gb.order)},
                "flat": gb.flatten()}
    schema_view = schema.export(schema_limit=10, per_schema_limit=4)
    return out, som_view, schema_view

# -------- Runner --------
def run_pipeline(json_path: str = "../assets/comprehension_results/simulation/simulation_read_contents.json",
                 llm_agent=None,
                 use_llm: bool = True,
                 wm_buffer: int = 8,
                 limit_episodes: Optional[int] = None,
                 start: int = 0,
                 log_every: int = 1,
                 verbose: str = "INFO",
                 p_store: float = 0.35,
                 allow_reparse: bool = True,
                 mode: str = "llm_schema",
                 som_limit: int = 12,
                 som_policy: str = "merge_topk",
                 parse_mode: str = "llm",
                 use_actual_index: bool = True,
                 ) -> Dict[str, Any]:

    logger = setup_logger(verbose)

    with open(json_path, "r", encoding="utf-8") as f:
        episodes = json.load(f)

    n_total = len(episodes)
    data_slice = episodes[start:] if limit_episodes is None else episodes[start: start + limit_episodes]
    logger.info(f"Loaded {n_total} episodes; processing {len(data_slice)} (start={start}).")

    results: Dict[str, Any] = {}
    agg_path = os.path.join("..", "assets", "comprehension_results", "simulation", "sim_ltm_gists.json")
    som_path = os.path.join("..", "assets", "comprehension_results", "simulation", "sim_ordered_gists.json")

    for idx, ep in enumerate(data_slice, start=1):
        t0 = time.time()
        ep_idx = int(ep.get("episode_index", idx - 1 + start))
        stim_idx = int(ep.get("stimulus_index", 0))
        cond = ep.get("time_condition", "60s")

        key = f"episode_{ep_idx}_stim_{stim_idx}_{cond}"
        tag = f"[{idx}/{len(data_slice)}] {key}"
        logger.info(f"{tag} BEGIN")

        cycles, som_view, schema_view = process_episode(
            ep=ep,
            llm_agent=llm_agent,
            logger=logger,
            som_limit=som_limit,
            som_policy=som_policy,
            p_store=p_store,
            use_actual_index=use_actual_index
        )

        results[key] = [asdict(c) for c in cycles]
        results[key + "__SOM"] = som_view
        results[key + "__SCHEMA_VIEW"] = schema_view
        results[key + "__LTM"] = {"p_store": p_store, "items": {}}

        dt = time.time() - t0
        logger.info(f"{tag} DONE in {dt:.2f}s | SOM sentences={len(som_view['per_sentence'])} | schemas={len(schema_view.get('schemas', []))}")

        # atomic aggregate updates after EACH episode
        agg = _load_json_or_empty(agg_path)
        agg[key] = results[key]
        agg[key + "__SOM"] = results[key + "__SOM"]
        agg[key + "__SCHEMA_VIEW"] = results[key + "__SCHEMA_VIEW"]
        agg[key + "__LTM"] = results[key + "__LTM"]
        _atomic_write_json(agg, agg_path)

        som_only = {k: v for k, v in agg.items() if k.endswith("__SOM")}
        _atomic_write_json(som_only, som_path)

        logger.info(f"{tag} aggregate updated → {agg_path} (and SOM-only → {som_path})")

    return results

if __name__ == "__main__":
    try:
        _ = run_pipeline()
    except FileNotFoundError:
        print("Default input JSON not found; call via your main.py")
