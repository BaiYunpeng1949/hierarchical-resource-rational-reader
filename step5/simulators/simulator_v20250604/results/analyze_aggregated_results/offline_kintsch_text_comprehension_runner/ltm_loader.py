
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

HEADER_RE = re.compile(r'^###\s*Episode\s*(\d+)\s*\|\s*Stimulus\s*(\d+)\s*\|\s*([0-9]+s)\s*(?:\((\d+)s\))?\s*$')

def _strip_markdown_bullets(text: str) -> str:
    """Return a lightly cleaned plaintext from a markdown bullet list."""
    lines = []
    for raw in text.splitlines():
        line = raw.rstrip()
        line = re.sub(r'^\s*[-*+]\s+', '', line)
        line = re.sub(r'^\s*\d+\.\s+', '', line)  # numbered lists
        if line.strip():
            lines.append(line)
    return "\n".join(lines).strip()

def parse_ltm_md(md_text: str) -> List[Dict]:
    """Parse sections like '### Episode i | Stimulus j | 30s (30s)' and their bullet blocks."""
    results: List[Dict] = []
    lines = md_text.splitlines()
    i = 0
    current = None
    buffer = []

    def flush():
        nonlocal current, buffer, results
        if current is None:
            return
        raw_block = "\n".join(buffer).strip()
        cleaned = _strip_markdown_bullets(raw_block)
        entry = {
            "episode_index": current["episode_index"],
            "stimulus_index": current["stimulus_index"],
            "time_condition": current["time_condition"],
            "total_time": current["total_time"],
            "raw_markdown": raw_block,
            "content": cleaned,
        }
        results.append(entry)
        buffer = []

    while i < len(lines):
        line = lines[i]
        m = HEADER_RE.match(line.strip())
        if m:
            flush()
            ep = int(m.group(1))
            stim = int(m.group(2))
            tcond = m.group(3)
            ttotal = int(m.group(4)) if m.group(4) is not None else None
            current = {
                "episode_index": ep,
                "stimulus_index": stim,
                "time_condition": tcond,
                "total_time": ttotal,
            }
        else:
            if current is not None:
                buffer.append(line)
        i += 1

    flush()
    return results

def load_ltm_from_md(path: Path) -> List[Dict]:
    text = Path(path).read_text(encoding="utf-8")
    return parse_ltm_md(text)

def build_ltm_index(entries: List[Dict]) -> Dict[Tuple[int,int,str], Dict]:
    idx: Dict[Tuple[int,int,str], Dict] = {}
    for e in entries:
        key = (e["episode_index"], e["stimulus_index"], e["time_condition"])
        idx[key] = e
    return idx

def select_ltm(entries_or_index, episode_index: int, stimulus_index: int, time_condition: str) -> Optional[Dict]:
    if isinstance(entries_or_index, dict):
        idx = entries_or_index
    else:
        idx = build_ltm_index(entries_or_index)
    return idx.get((episode_index, stimulus_index, time_condition))

def build_prompt_from_ltm(entry: Dict, header: str = "Long-term Memory (LTM) Summary") -> str:
    if entry is None:
        return ""
    parts = [
        f"## {header}",
        f"- Episode: {entry['episode_index']}  Stimulus: {entry['stimulus_index']}  Time: {entry['time_condition']}",
        "",
        entry["content"],
    ]
    return "\n".join(parts).strip()
