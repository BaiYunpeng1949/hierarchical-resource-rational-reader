# ci_schema_pipeline.py
from __future__ import annotations
import json, re, math, itertools
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Iterable, Optional
import numpy as np

import logging, time

def setup_logger(level: str = "INFO") -> logging.Logger:  # NEW
    logger = logging.getLogger("ci_pipeline")
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger

# --- If you saved ci_model.py from yesterday, you can import it.
# from ci_model import Proposition, Node, CINetwork, build_network, integrate
# For self-containment, I inline minimal structures again:

@dataclass(frozen=True)
class Proposition:
    predicate: str
    args: Tuple[str, ...]
    sent_id: int
    span: Tuple[int, int]

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
    edges: List[Tuple[int,int,float]] = field(default_factory=list)
    index: Dict[str,int] = field(default_factory=dict)

    def add_node(self, node: Node) -> int:
        if node.key in self.index: return self.index[node.key]
        i = len(self.nodes); self.nodes.append(node); self.index[node.key] = i; return i
    def add_edge(self, i:int, j:int, w:float):
        if i!=j and w!=0: self.edges.append((i,j,w))

# --- Utils

def _clean_join(words: List[str]) -> str:
    # your sampled list sometimes has "" and tokens with commas/periods attached
    toks = [w for w in words if w and isinstance(w, str)]
    s = " ".join(toks)
    # light fixes of obvious corruption (optional)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _heuristic_propositions(sent: str, sent_id: int) -> List[Proposition]:
    # extremely lightweight SVO-ish propositions as a robust fallback
    toks = re.findall(r"[A-Za-z0-9%'-]+|[.,!?;:]", sent)
    if not toks:
        return [Proposition("mention",(sent.lower(),),sent_id,(0,len(sent)))]
    props = []
    # naive verb guess and neighbors
    verbs = [i for i,t in enumerate(toks) if re.match(r".*(ed|ing|s)$", t.lower()) or t.lower() in
             {"is","are","was","were","be","am","been","has","have","had","will"}]
    for i in verbs:
        v = toks[i].lower()
        # subject left
        subj = next((toks[j].lower() for j in range(i-1,-1,-1) if toks[j].isalnum()), None)
        # object right
        obj = next((toks[j].lower() for j in range(i+1,len(toks)) if toks[j].isalnum()), None)
        if subj and obj:
            props.append(Proposition(v,(subj,obj),sent_id,(0,len(sent))))
    if not props:
        head = next((t.lower() for t in toks if t.isalnum()), "mention")
        props.append(Proposition("mention",(head,),sent_id,(0,len(sent))))
    # dedupe
    out, seen = [], set()
    for p in props:
        sig = (p.predicate,p.args,p.sent_id)
        if sig not in seen: out.append(p); seen.add(sig)
    return out

# --- GPT interface hook

class PropositionParser:
    def __init__(self, llm_agent=None, role="", use_llm=True, logger: Optional[logging.Logger]=None):
        self.llm = llm_agent
        self.role = role
        self.use_llm = (llm_agent is not None) and use_llm
        self.logger = logger
        self.last_method = "heuristic"

    def parse(self, sent: str, sent_id: int) -> List[Proposition]:
        self.last_method = "heuristic"  # default
        if self.use_llm:
            props = self._parse_with_llm(sent, sent_id)
            if props:
                self.last_method = "llm"
                return props
        return _heuristic_propositions(sent, sent_id)

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
                    if not m: continue
                    pred = m.group(1).lower()
                    arg_str = m.group(2).strip()
                    # split top-level commas (no nested parse here; flatten)
                    args = tuple(a.strip().lower() for a in re.split(r"\s*,\s*", arg_str) if a.strip())
                    if pred:
                        props.append(Proposition(pred, args, sent_id, (0,len(sent))))
            # dedupe
            uniq, seen = [], set()
            for p in props:
                sig = (p.predicate,p.args,p.sent_id)
                if sig not in seen: uniq.append(p); seen.add(sig)
            return uniq
        except Exception as e:
            if self.logger:
                self.logger.debug(f"LLM parse failed, falling back. Reason: {e}")
            return None

# --- Build & integrate (Kintsch-style constraint satisfaction)

def build_network(props: List[Proposition],
                  coarg_w: float=0.7, copred_w: float=0.3,
                  cross_sent_decay: float=0.85) -> CINetwork:
    net = CINetwork()
    ids = []
    for p in props:
        i = net.add_node(Node(key=p.signature(), prop=p))
        ids.append(i)
    for i,j in itertools.combinations(range(len(net.nodes)),2):
        p1, p2 = net.nodes[i].prop, net.nodes[j].prop
        dist = abs(p1.sent_id - p2.sent_id)
        dist_factor = cross_sent_decay ** dist
        arg_overlap = len(set(p1.args)&set(p2.args))
        w1 = coarg_w * arg_overlap * dist_factor
        w2 = copred_w * (1.0 if p1.predicate==p2.predicate else 0.0) * dist_factor
        w = w1 + w2
        if w>0: net.add_edge(i,j,w)
    return net

def integrate(net: CINetwork, iters:int=12, leak:float=0.10, beta:float=1.0) -> None:
    n = len(net.nodes)
    if n==0: return
    W = np.zeros((n,n))
    for i,j,w in net.edges:
        W[i,j]+=w; W[j,i]+=w
    # init: earlier sentences slightly higher
    sids = np.array([nd.prop.sent_id for nd in net.nodes], float)
    base = 1.0 + (sids.max() - sids)/(max(1.0, sids.max()))*0.2
    a = base/base.sum()
    for _ in range(iters):
        a = (1-leak)*a + beta*(W@a)
        a = np.maximum(a,0)
        s=a.sum()
        if s>0: a/=s
    for i,val in enumerate(a): net.nodes[i].activation = float(val)

# --- Dynamic schema + macro rules (deletion/generalization/construction)

@dataclass
class Schema:
    """Lightweight evolving schema = concept weights."""
    weights: Dict[str, float] = field(default_factory=dict)
    decay: float = 0.90
    learn_rate: float = 0.25

    def update_from_props(self, props: List[Proposition], strength: float=1.0):
        # bump concepts appearing as predicate/args
        seen = {}
        for p in props:
            seen[p.predicate] = seen.get(p.predicate,0)+1
            for a in p.args:
                seen[a] = seen.get(a,0)+1
        # decay
        for k in list(self.weights.keys()):
            self.weights[k] *= self.decay
            if self.weights[k] < 1e-5:
                del self.weights[k]
        # learn
        for k,cnt in seen.items():
            self.weights[k] = self.weights.get(k,0.0) + self.learn_rate*strength*cnt

    def relevance(self, prop: Proposition) -> float:
        # simple overlap score with schema
        score = self.weights.get(prop.predicate,0.0) + sum(self.weights.get(a,0.0) for a in prop.args)
        return score

def macroselect_gist(props: List[Proposition], schema: Schema,
                     buffer_size:int=5, rel_thresh:float=0.10) -> List[Proposition]:
    """
    Apply Kintsch-style macroprocess control:
      - Deletion: drop propositions not relevant to schema nor needed as interpretation conditions.
      - Generalization: (stub) collapse repeated predicates on same arguments.
      - Construction: (stub) compose 'global' proposition for clusters.
    This minimal version prioritizes schema relevance + referential connectivity. :contentReference[oaicite:1]{index=1}
    """
    if not props: return []
    # relevance
    scored = [(p, schema.relevance(p)) for p in props]
    scored.sort(key=lambda x: x[1], reverse=True)
    # keep top buffer_size, but ensure connectivity (referential coherence) :contentReference[oaicite:2]{index=2}
    keep = []
    argset = set()
    for p,score in scored:
        if len(keep) < buffer_size or score >= rel_thresh:
            # prefer props that share arguments with kept set (coherence)
            share = (len(argset & set(p.args))>0) or (not keep)
            if share or len(keep) < buffer_size//2:
                keep.append(p)
                argset |= set(p.args)
    # (Optional) construction/generalization stubs:
    # collapse duplicates with same predicate/args
    uniq, seen = [], set()
    for p in keep:
        sig = (p.predicate,p.args)
        if sig not in seen: uniq.append(p); seen.add(sig)
    return uniq

# --- Visit (re-read) gain

def apply_visit_gain(props: List[Proposition], visit_count: int, base_gain: float=0.05) -> List[Proposition]:
    """
    Increase schema learning strength for repeated visits to the same sentence.
    Each revisit adds a small multiplicative gain to 'strength' used in schema.update_from_props.
    Empirically matches: multiple processing → higher recall probability. :contentReference[oaicite:3]{index=3}
    """
    strength = 1.0 + base_gain*max(0, visit_count-1)
    # Return same props; caller will pass 'strength' to schema.update_from_props
    return props, strength

# --- Main: process a single episode

@dataclass
class CycleOutput:
    step: int
    sent_id: int
    sentence_text: str
    propositions: List[str]
    gist_selected: List[str]
    schema_snapshot: Dict[str,float]

def process_episode(ep: Dict,
                    parser: PropositionParser,
                    schema: Optional[Schema]=None,
                    wm_buffer:int=5,
                    logger: Optional[logging.Logger]=None,
                    episode_tag: str="",
                    log_every: int=1
                    ) -> List[CycleOutput]:
    if schema is None: schema = Schema()
    # track revisits per sentence
    visit_counter: Dict[int,int] = {}
    out: List[CycleOutput] = []
    total_steps = len(ep["text_reading_logs"])

    t_ep0 = time.perf_counter()
    for k, log in enumerate(ep["text_reading_logs"], start=1):
        t0 = time.perf_counter()       

        sent_id = log["current_sentence_index"]
        visit_counter[sent_id] = visit_counter.get(sent_id,0)+1
        sent = _clean_join(log["sampled_words_in_sentence"])
        if not sent:
            continue

        # parse micro-props
        props = parser.parse(sent, sent_id=sent_id)

        # visit-based gain
        props, strength = apply_visit_gain(props, visit_counter[sent_id])

        # update schema from full microstructure, then macroselect gist for LTM
        schema.update_from_props(props, strength=strength)
        gist = macroselect_gist(props, schema, buffer_size=wm_buffer)

        # build/integrate a local CI network (optional: per-cycle coherence update)
        net = build_network(props)
        integrate(net)

        out.append(CycleOutput(
            step=log["step"],
            sent_id=sent_id,
            sentence_text=sent,
            propositions=[p.signature() for p in props],
            gist_selected=[p.signature() for p in gist],
            schema_snapshot=dict(sorted(schema.weights.items(), key=lambda x:-x[1])[:32]) # top schema terms
        ))

        # --- PROGRESS LOGGING (every N steps and at the end)
        if logger and (k % log_every == 0 or k == total_steps):
            top_schema = ", ".join([kv[0] for kv in sorted(schema.weights.items(), key=lambda x:-x[1])[:3]]) or "-"
            dt = time.perf_counter() - t0
            logger.info(
                f"{episode_tag} step {k}/{total_steps} | sent={sent_id} visit={visit_counter[sent_id]} | "
                f"{len(props)} props ({parser.last_method}), gist={len(gist)} | "
                f"schema↟ [{top_schema}] | {dt:.2f}s"
            )

    if logger:
        logger.info(f"{episode_tag} DONE in {time.perf_counter() - t_ep0:.2f}s")  
    return out

# --- Entry: run over the JSON

def run_pipeline(json_path: str,
                 llm_agent=None,
                 llm_role:str="You are a careful proposition parser per Kintsch.",
                 use_llm: bool=True,
                 wm_buffer:int=5,
                 limit_episodes: Optional[int]=None,
                 start: int=0,
                 log_every: int=1,
                 verbose: str="INFO") -> Dict:
    logger = setup_logger(verbose)

    with open(json_path,"r",encoding="utf-8") as f:
        data = json.load(f)

    # slice the episodes                              
    if limit_episodes is not None:
        data_slice = data[start:start+limit_episodes]
    else:
        data_slice = data[start:]
    
    logger.info(f"Loaded {len(data)} episodes; processing {len(data_slice)} (start={start}).")  

    parser = PropositionParser(llm_agent=llm_agent, role=llm_role, use_llm=use_llm)
    results = {}

    for idx, ep in enumerate(data_slice, start=1):
        key = f"episode_{ep['episode_index']}_stim_{ep['stimulus_index']}_{ep['time_condition']}"
        tag = f"[{idx}/{len(data_slice)}] {key}"
        logger.info(f"{tag} BEGIN") 
        schema = Schema()
        cycles = process_episode(ep, parser, schema=schema, wm_buffer=wm_buffer,
                                 logger=logger, episode_tag=tag, log_every=log_every)
        results[key] = [c.__dict__ for c in cycles]

    return results


# TODO: add running logs, so we could know where are we when generating, to avoid wasting too much tokens of our api