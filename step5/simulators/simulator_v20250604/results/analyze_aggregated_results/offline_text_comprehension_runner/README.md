# Construction–Integration (CI) Text Comprehension Pipeline

This repository implements a **Kintsch-style Construction–Integration (CI)** pipeline, adapted to sentence-level cycles with an evolving **schema controller**, a **working-memory (WM) buffer**, and a lightweight **LTM (long-term memory) store**. It parses micro-structural propositions per cycle (sentence), performs constraint–satisfaction integration for local coherence, applies schema-controlled macrorules to select a per-cycle gist (**WM**), and accumulates an LTM summary with an explicit **multiple-processing advantage**.

> Key files:
> - `ci_schema_pipeline.py` – main pipeline (schema + WM + LTM + logs).
> - `gpt-api.py` – your LLM wrapper (optional; pipeline falls back to heuristic parsing).
> - `simulation_read_contents.json` – episode-wise input logs of what the agent read.
> - (optional runner) `main.py` – simple CLI runner (see Usage).

---

## 1. Conceptual overview

**Cycle** = one sentence read-in. For each cycle we:
1. **Construct** a microstructure: parse the sentence into **propositions** `predicate(arg1, arg2, ...)`.
2. **Integrate** locally using a constraint–satisfaction network (co-argument + co-predicate links; iterative activation).
3. **Schema update**: decay old weights; learn from concepts seen in this cycle (predicate + args), amplified by **re-read gain** for revisits.
4. **Macroselection (WM gist)**: score propositions by **schema relevance** and **prefer** those that **share arguments** (referential coherence), then keep up to `wm_buffer` items.
5. **LTM update**: each proposition selected into WM increments an LTM entry with a **visit counter** and a simple **recall proxy** `1 - (1 - p_store)^visits`.

This operationalizes Kintsch & van Dijk’s account:
- **Working memory constraint**: a small buffer carries **n** propositions across cycles to enforce coherent bridging.
- **Schema-controlled macrorules**: a dynamic schema determines relevance; **Deletion/Generalization/Construction** decide what survives as gist (we implement Deletion + coherence; Gen/Constr hooks are provided).
- **Multiple processing → better recall**: re-reading increases learning **strength**, and repeated WM selection raises LTM **visit counts** (and recall proxy).

---

## 2. Input data format

`simulation_read_contents.json` is a list of episodes. Each episode minimally contains:

```jsonc
[
  {
    "episode_index": 0,
    "stimulus_index": 0,
    "time_condition": "30s",
    "total_time": 30,
    "text_reading_logs": [
      {
        "step": 1,
        "current_sentence_index": 0,
        "num_words_in_sentence": 18,
        "actual_reading_sentence_index": 0,
        "sampled_words_in_sentence": ["Readers", "skip", "more", "words", "...", "."]
      },
      ...
    ]
  },
  ...
]
```

Only `sampled_words_in_sentence` is strictly required by the parser; the rest supports logging and analysis.

---

## 3. Pipeline API (high level)

```python
from ci_schema_pipeline import run_pipeline
from gpt-api import LLMAgent  # optional

llm = LLMAgent(model_name="gpt-4o")   # or None to disable LLM parsing

results = run_pipeline(
    json_path="simulation_read_contents.json",
    llm_agent=llm,           # set to None or use_llm=False to use heuristic parser
    use_llm=True,
    wm_buffer=8,             # WM capacity (per-cycle gist size)
    limit_episodes=1,        # process first N episodes (None = all)
    start=0,                 # start offset
    log_every=1,             # print progress every k cycles
    verbose="INFO",          # or "DEBUG" for detailed WM/LTM lines
    p_store=0.35             # per-cycle consolidation probability for recall proxy
)
```

A minimal CLI runner is included in earlier examples (`run_ci_from_json.py`), but any short script that calls `run_pipeline(...)` and writes `results` to disk is fine.

---

## 4. Design details & parameters

### 4.1 Parsing
- **LLM-assisted**: `PropositionParser` calls `llm.get_micro_structural_propositions(...)` with a strict output format; we parse lines like `predicate(arg1, arg2)`.  
- **Heuristic fallback**: an SVO-ish extractor (`_heuristic_propositions`) to keep cycles robust if the LLM is disabled.
- **Proposition cache**: Cache sentence parses so the same sentence (within a run) always yields the exact same propositions, preventing GPT variation on re-reads or repeated mentions.

**Variables**
- `llm_agent`: your `LLMAgent` instance from `gpt-api.py`.
- `use_llm` (bool): switch between LLM and heuristic.
- `llm_role` (str): optional role text passed to your agent (default already set in pipeline).

### 4.2 Integration (CI step)
- Build a small network where nodes are propositions.
- Edges: `coarg_w * (#shared_args)` + `copred_w * 1[predicates match]`.
- Cross-sentence decay for links, `cross_sent_decay ** distance`.
- Iterate activations: `a <- (1 - leak)*a + beta * W a` for `iters` rounds, with non-negativity and normalization.

**Variables**
- `coarg_w` (default 0.7): weight for co-argument overlap.
- `copred_w` (0.3): weight for predicate match.
- `cross_sent_decay` (0.85): link decay with sentence distance.
- `iters` (12): integration iterations.
- `leak` (0.10): activation leak.
- `beta` (1.0): integration gain.

### 4.3 Schema update (dynamic control)
- **Decay**: `weights[k] *= decay` each cycle.
- **Learn**: `weights[k] += learn_rate * strength * count`, where `k` spans predicate and all arguments seen.
- **Re-read gain**: if a sentence is revisited `v` times, `strength = 1 + base_gain*(v-1)` (default `base_gain=0.05`).

**Variables**
- `decay` (0.90): schema forgetting rate per cycle.
- `learn_rate` (0.25): how strongly concepts are reinforced when encountered.
- `base_gain` (0.05): extra learning per additional visit to the same sentence (re-read).

### 4.4 Macroprocess (WM gist selection)
- **Relevance score**: `Schema.relevance(prop) = w[predicate] + Σ w[arg]`.
- **Coherence preference**: prefer propositions that **share arguments** with already chosen items (referential coherence).
- **WM capacity**: keep at most `wm_buffer` propositions for this cycle’s WM.

**Variables**
- `wm_buffer` (5–8 typical): per-cycle WM capacity.
- `rel_thresh` (0.10 internal default): soft keep threshold (minor role if `wm_buffer` is reached).

**Output per step (cycle)**
- `wm_kept`: exact propositions in WM **this** cycle.
- `wm_retained`: WM items that **carried over** from the previous cycle.
- `wm_added`: newly added to WM this cycle.
- `wm_dropped`: those that were in WM previously but not selected now.

### 4.5 LTM store & recall proxy
- Each time a proposition is selected into WM, we **update LTM**:
  - `visits += 1`, `first_step`, `last_step`, accumulate `total_strength` and `last_relevance`.
  - `est_recall = 1 - (1 - p_store) ** visits` as a simple recall probability proxy.
- The store is episode-local (reset each episode).

**Variables**
- `p_store` (0.35): per-cycle consolidation probability used by the recall proxy.

**Output**
- `ltm_updates` per step: list of `{ "proposition", "visits", "est_recall", "last_relevance" }` for items selected this cycle.
- Episode-level summary is saved under `...__LTM` with the entire store.

---

## 5. Outputs (JSON schema)

`run_pipeline(...)` returns a dict keyed by episode. For each `episode_key` (e.g., `episode_0_stim_0_30s`) you get a list of **cycle records**:

```jsonc
{
  "episode_0_stim_0_30s": [
    {
      "step": 1,
      "sent_id": 0,
      "sentence_text": "Readers skip more words ...",
      "propositions": ["skip(readers, words)", "when(time, scarce)", "..."],
      "wm_kept": ["skip(readers, words)", "..."],
      "wm_retained": [],
      "wm_added": ["skip(readers, words)"],
      "wm_dropped": [],
      "ltm_updates": [
        {"proposition": "skip(readers, words)", "visits": 1, "est_recall": 0.35, "last_relevance": 0.42}
      ],
      "schema_snapshot": {"readers": 0.22, "skip": 0.21, "words": 0.20, "time": 0.14, "scarce": 0.12}
    },
    ...
  ],

  "episode_0_stim_0_30s__LTM": {
    "p_store": 0.35,
    "items": {
      "skip(readers, words)": {
        "visits": 3, "first_step": 1, "last_step": 12,
        "total_strength": 3.10, "last_relevance": 0.55, "est_recall": 0.72
      },
      "...": { ... }
    }
  }
}
```

You can write this to disk as `ci_gist_outputs.json` with a small runner:
```python
import json
from gpt-api import LLMAgent
from ci_schema_pipeline import run_pipeline

llm = LLMAgent(model_name="gpt-4o")
results = run_pipeline("simulation_read_contents.json", llm_agent=llm, use_llm=True, wm_buffer=8,
                       limit_episodes=1, log_every=1, verbose="INFO", p_store=0.35)
with open("ci_gist_outputs.json","w",encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

---

## 6. Logging

Use `verbose="DEBUG"` for detailed internals:
- **INFO**: episode progress; per-step counts; WM size & deltas; number of LTM updates; and top LTM items.
- **DEBUG**: exact WM lists (now/retained/added/dropped) and compact per-prop LTM lines (`proposition (visits, est_recall~, rel~)`).

Example INFO line:
```
[12:01:33] INFO [1/1] episode_0_stim_0_30s step 7/30 | sent=2 visit=1 | 5 props (llm),
WM=5 [ret 3, +2, -2] | LTM+=5 (top: skip(readers, words), install(stations, new)) | 0.18s
```

---

## 7. Replication steps

1) **Place data & code**  
   - Ensure `simulation_read_contents.json` is in the working dir (or pass a full path).  
   - Confirm `ci_schema_pipeline.py` and `gpt-api.py` are in your PYTHONPATH (same folder is fine).

2) **Configure LLM (optional)**  
   - Edit `gpt-api.py` to set your model and key (or use environment variables).  
   - To disable LLM parsing, call with `use_llm=False` (heuristic fallback is deterministic).

3) **Run**  
   - Minimal Python script (see code above), or use your CLI runner with flags. Note that `--episodes` number determines how many trials to cover. Set `1` for a smoking check.
   - CI schema integration on:
     ```bash
     python main.py --episodes 1 --log_every 1 --mode ci_schema --som_policy merge_topk --verbose INFO
     ```
    - CI schema integration off (for testing and comparisons):
      ```bash
      python main.py --episodes 1 --log_every 1 --mode none --som_policy merge_topk --verbose INFO
      ```
    - CI schema integartion off, and use the raw sentences as propositions (other wise, use LLM to process, choose `--parse_mode llm`). Do this for comparison tets. All 27 trials (3 conditions * 9 stimuli)
      ```bash
      python main.py --episodes 27 --log_every 1 --parse_mode raw --mode none --som_policy merge_topk --verbose INFO
      ```
    - CI schema integration off, use 'Facet Summaries' as propositions, instead of strict A(B, C)'s kintsch's propositions, which was over-fragmenting and loses too much information.
      ```bash
      python main.py --episodes 3 --log_every 1 --parse_mode facets --mode none --som_policy merge_topk --verbose INFO
      ```

4) **Inspect results**  
   - Open the generated JSON; check `wm_*` fields and `ltm_updates` for each step.  
   - For an aggregated view, count LTM `visits` per proposition (higher visits ≈ better recall).

5) **(Optional) Performance tips**  
   - Reduce `--episodes` and/or increase `--log_every` during development.  
   - Cache LLM outputs (by sentence text) if your dataset has re-reads or repeated stimuli.  
   - Set `use_llm=False` when you only want to debug selection/integration logic quickly.

---

## 8. Interpreting WM, Schema, and LTM

- **WM buffer** is the **bridging set** for the next cycle; it is *not* the only content written to LTM, but in this implementation, **only WM-selected items** update the LTM store (to reflect schema-controlled macroselection).  
- **Schema snapshot** provides the **conceptual focus** after the cycle; it is *not* the gist. Use `wm_kept` (and deltas) for the exact cycle gist.  
- **LTM store** tracks repeated selection (`visits`) and a recall proxy (`est_recall`). This captures Kintsch’s **multiple processing advantage** explicitly.

---

## 9. Extending the model

- **Generalization/Construction rules**: add a small verb-frame lexicon to collapse micro-props into macro-props (e.g., `{install(stations, new), add(bikes, fleet)} → expand(bikeshare)`).
- **Slot-based schema**: introduce slots (`Stakeholders, Policy, Infrastructure, Costs, Benefits, Risks`) and route propositions to slots (via a small rule map or a classifier prompt). Store gist as a dict `{slot -> [props]}`; score relevance as `slot_weight + concept_weight`.
- **Time pressure**: modulate `p_store`, `decay`, and/or `wm_buffer` by `time_condition` to capture strategic changes under constraints.
- **Deterministic runs**: turn off LLM (`use_llm=False`), or seed your agent if it supports deterministic sampling.

---

## 10. Variables summary

| Variable | Where | Meaning / Effect |
|---|---|---|
| `wm_buffer` | `run_pipeline`, `macroselect_gist` | WM capacity per cycle (number of propositions kept in WM/gist). |
| `coarg_w` | `build_network` | Weight for co-argument overlap edges. |
| `copred_w` | `build_network` | Weight for co-predicate match edges. |
| `cross_sent_decay` | `build_network` | Edge decay by sentence distance. |
| `iters` | `integrate` | CI integration iterations. |
| `leak` | `integrate` | Activation leak per iteration. |
| `beta` | `integrate` | Integration gain. |
| `decay` | `Schema` | Schema weight decay per cycle. |
| `learn_rate` | `Schema` | Learning rate for concepts seen in the cycle. |
| `base_gain` | `apply_visit_gain` | Extra learning per additional re-read of a sentence. |
| `p_store` | `LTMStore` | Per-cycle consolidation parameter used for recall proxy. |
| `limit_episodes`, `start` | `run_pipeline` | Process only a slice of episodes. |
| `log_every`, `verbose` | `run_pipeline` | Logging cadence and verbosity. |
| `use_llm`, `llm_agent`, `llm_role` | `PropositionParser` | Parser settings (LLM vs heuristic). |

---

## 11. Changelog (2025‑08‑20)

- Added **explicit WM tracking** (`wm_kept`, `wm_retained`, `wm_added`, `wm_dropped`).
- Added **LTM store** with per-step `ltm_updates`, visit counts, and recall proxy.
- Enriched logging: WM sizes/deltas; top LTM items; detailed DEBUG dumps.
- Preserved compatibility with prior outputs (fields extended, not removed).

---

## 12. Reference

- Kintsch, W., & van Dijk, T. A. (1978). *Toward a model of text comprehension and production*. **Psychological Review**, 85(5), 363–394.
