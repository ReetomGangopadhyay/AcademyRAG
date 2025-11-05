import os
from typing import List, Dict, Any

from openai import OpenAI
from .prompts import ANSWER_SYSTEM, SUMMARY_SYSTEM, QUIZ_SYSTEM

def _client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _format_context(docs: List[Dict[str, Any]]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        meta = d.get("metadata", {})
        title = meta.get("doc_title", "Document")
        page = meta.get("page")
        slide = meta.get("slide_title")
        loc = f"(p.{page})" if page is not None else ""
        if slide:
            loc = f"{loc} — {slide}" if loc else slide
        parts.append(f"[{i}] {title} {loc}\n{d['text']}")
    return "\n\n".join(parts)

def _citations(docs: List[Dict[str, Any]]):
    cits = []
    for d in docs:
        meta = d.get("metadata", {})
        cits.append({
            "doc_title": meta.get("doc_title", "Document"),
            "page": meta.get("page"),
            "slide_title": meta.get("slide_title"),
            "source_path": meta.get("source_path")
        })
    return cits

def _chat(system: str, user: str) -> str:
    prov = os.getenv("ACADEMYRAG_LLM_PROVIDER", "openai")
    if prov != "openai":
        raise RuntimeError("Only OpenAI chat supported in this MVP. Set ACADEMYRAG_LLM_PROVIDER=openai and OPENAI_API_KEY.")
    cl = _client()
    msg = cl.chat.completions.create(
        model=os.getenv("ACADEMYRAG_LLM_MODEL", "gpt-4o-mini"),
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":user}
        ],
        temperature=0.2,
    )
    return msg.choices[0].message.content

def generate_answer(question: str, docs: List[Dict[str, Any]]):
    ctx = _format_context(docs)
    user = f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx}"
    text = _chat(ANSWER_SYSTEM, user)
    return {"text": text, "citations": _citations(docs)}

def generate_summary(topic: str, docs: List[Dict[str, Any]]):
    ctx = _format_context(docs)
    user = f"TOPIC: {topic}\n\nCONTEXT:\n{ctx}"
    text = _chat(SUMMARY_SYSTEM, user)
    return {"text": text, "citations": _citations(docs)}

def generate_quiz(topic: str, docs: List[Dict[str, Any]]):
    ctx = _format_context(docs)
    user = f"TOPIC: {topic}\n\nCONTEXT:\n{ctx}"
    text = _chat(QUIZ_SYSTEM, user)
    # Expecting JSON-like but model returns text; attempt to parse simply
    import json, re
    try:
        # Try to extract JSON block
        jtxt = re.search(r"\{[\s\S]*\}\s*$", text).group(0)
        data = json.loads(jtxt)
        return data
    except Exception:
        # Fallback: return raw text packaged
        return {"raw": text, "questions": []}
