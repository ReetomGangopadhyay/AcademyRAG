"""
Evaluation utilities for AcademyRAG.

Metrics:
- Recall@k, Precision@k
- MRR (Mean Reciprocal Rank)
- MAP (Mean Average Precision)
- nDCG@k
- Faithfulness (lexical grounding score of answer to retrieved context)

Dataset format (JSONL):
Each line is a JSON object:
{
  "query": "What are common cost drivers?",
  "gold": [
    {"doc_title": "value_chain_playbook.md", "page": null},
    {"doc_title": "cost_drivers_101.md", "page": null}
  ],
  "k": 6  # optional per-example override
}

Gold matching rule:
A retrieved chunk counts as a hit if its metadata matches any gold item:
- doc_title must match; and if gold.page is not null, page must match too.
"""

from __future__ import annotations
import json, math, argparse, os, re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter

from .retrieve import retrieve_with_rerank
from .generate import generate_answer

def _normalize_text(s: str) -> List[str]:
    if not s:
        return []
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t]
    return toks

def _jaccard(a: str, b: str) -> float:
    A, B = set(_normalize_text(a)), set(_normalize_text(b))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def _match_retrieved_to_gold(retrieved: List[Dict[str, Any]],
                             gold: List[Dict[str, Any]]) -> List[int]:
    """
    Returns a binary list hits where hits[i] == 1 if retrieved[i] matches any gold item.
    Match on doc_title, and (if provided) page must match.
    """
    hits = []
    for d in retrieved:
        meta = d.get("metadata", {})
        r_title = meta.get("doc_title")
        r_page = meta.get("page", None)
        ok = 0
        for g in gold:
            g_title = g.get("doc_title")
            g_page  = g.get("page", None)
            if r_title == g_title:
                if g_page is None or g_page == r_page:
                    ok = 1
                    break
        hits.append(ok)
    return hits

def precision_at_k(hits: List[int], k: int) -> float:
    k = min(k, len(hits))
    if k == 0: return 0.0
    return sum(hits[:k]) / float(k)

def recall_at_k(hits: List[int], k: int, total_relevant: int) -> float:
    if total_relevant <= 0: 
        return 0.0
    return sum(hits[:k]) / float(total_relevant)

def reciprocal_rank(hits: List[int]) -> float:
    for i, h in enumerate(hits, start=1):
        if h:
            return 1.0 / i
    return 0.0

def average_precision(hits: List[int]) -> float:
    # AP = mean of precision at ranks where a relevant item occurs
    num_rel = sum(hits)
    if num_rel == 0:
        return 0.0
    cum = 0.0
    found = 0
    for i, h in enumerate(hits, start=1):
        if h:
            found += 1
            cum += found / i
    return cum / num_rel

def dcg_at_k(gains: List[int], k: int) -> float:
    k = min(k, len(gains))
    return sum(gains[i] / math.log2(i+2) for i in range(k))

def ndcg_at_k(hits: List[int], k: int) -> float:
    dcg = dcg_at_k(hits, k)
    ideal = sorted(hits, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

def faithfulness_score(answer_text: str, retrieved_texts: List[str]) -> float:
    """
    Simple lexical grounding: compute max Jaccard overlap between the answer
    and each retrieved chunk; return the average of the top-3 overlaps.
    Range: [0,1]. Higher = more grounded.
    """
    if not answer_text or not retrieved_texts:
        return 0.0
    scores = [_jaccard(answer_text, ctx) for ctx in retrieved_texts]
    scores.sort(reverse=True)
    top = scores[:3] if len(scores) >= 3 else scores
    if not top: 
        return 0.0
    return sum(top) / len(top)

def evaluate_query(query: str,
                   gold: List[Dict[str, Any]],
                   k: int = 6,
                   judge_answer: bool = True) -> Dict[str, float]:
    """
    Retrieve, compute ranking metrics, optionally generate answer and compute faithfulness.
    """
    retrieved = retrieve_with_rerank(query, top_k=k)
    hits = _match_retrieved_to_gold(retrieved, gold)
    total_relevant = len(gold)

    metrics = {
        "P@k": precision_at_k(hits, k),
        "R@k": recall_at_k(hits, k, total_relevant),
        "MRR": reciprocal_rank(hits),
        "MAP": average_precision(hits),
        "nDCG@k": ndcg_at_k(hits, k),
    }

    if judge_answer:
        # Build answer from retrieved context and score grounding
        ans = generate_answer(query, retrieved)
        answer_text = ans.get("text", "") or ""
        retrieved_texts = [d.get("text", "") for d in retrieved]
        metrics["faithfulness"] = faithfulness_score(answer_text, retrieved_texts)
        metrics["answer_len"] = len(answer_text.split())
    return metrics

def evaluate_dataset(jsonl_path: str,
                     default_k: int = 6,
                     judge_answer: bool = True) -> Dict[str, Any]:
    """
    Evaluate a JSONL dataset; returns macro-averaged metrics and per-example results.
    """
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            q = ex["query"]
            gold = ex.get("gold", [])
            k = int(ex.get("k", default_k))
            m = evaluate_query(q, gold, k=k, judge_answer=judge_answer)
            results.append({"query": q, "k": k, **m})

    # Aggregate
    agg = defaultdict(float)
    for r in results:
        for key, val in r.items():
            if key in ("query", "k"):
                continue
            agg[key] += float(val)

    n = max(1, len(results))
    macro = {k: v / n for k, v in agg.items()}
    return {"macro": macro, "results": results}

def _fmt_table(rows: List[Tuple[str, float]]) -> str:
    name_w = max(len(n) for n, _ in rows)
    lines = []
    for n, v in rows:
        lines.append(f"{n.ljust(name_w)} : {v:0.4f}")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Evaluate AcademyRAG on a JSONL dataset.")
    ap.add_argument("--data", required=True, help="Path to eval.jsonl")
    ap.add_argument("--k", type=int, default=6, help="Default top-k for retrieval")
    ap.add_argument("--no-answer", action="store_true", help="Skip answer generation & faithfulness")
    args = ap.parse_args()

    report = evaluate_dataset(args.data, default_k=args.k, judge_answer=(not args.no_answer))
    macro = report["macro"]
    print("\n== Macro Averages ==")
    rows = [(k, macro[k]) for k in sorted(macro.keys())]
    print(_fmt_table(rows))

    # Optionally write detailed results
    out_path = os.path.splitext(args.data)[0] + ".results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote detailed results → {out_path}")

if __name__ == "__main__":
    main()
