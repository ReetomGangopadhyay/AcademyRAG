import os
import streamlit as st
from dotenv import load_dotenv

from rag.ingest import ingest_path
from rag.retrieve import retrieve_with_rerank
from rag.store import get_store  # kept for future use (ensures DB path exists)

load_dotenv()
LLM_PROVIDER = os.getenv("ACADEMYRAG_LLM_PROVIDER", "openai").lower()

# Import generation utils only if we actually have an LLM provider
if LLM_PROVIDER != "none":
    from rag.generate import generate_answer, generate_summary, generate_quiz

st.set_page_config(page_title="AcademyRAG", layout="wide")
st.title("AcademyRAG - Learning Assistant")

# If no LLM, let the user know (retrieval-only mode)
if LLM_PROVIDER == "none":
    st.info(
        "Running in **no-LLM mode**: embeddings & retrieval are local. "
        "The app will show top matches with citations instead of generated answers/summaries/quizzes. "
        "Set `ACADEMYRAG_LLM_PROVIDER=openai` (and an API key) to enable generation."
    )

# Sidebar controls
db_dir = os.getenv("ACADEMYRAG_DB_DIR", "./data/index")
chunk_size = int(os.getenv("ACADEMYRAG_CHUNK_SIZE", "900"))
chunk_overlap = int(os.getenv("ACADEMYRAG_CHUNK_OVERLAP", "120"))

st.sidebar.header("Indexer")
uploaded = st.sidebar.file_uploader(
    "Upload docs (PDF, PPTX, MD, TXT)",
    type=["pdf", "pptx", "md", "txt"],
    accept_multiple_files=True
)
if uploaded:
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    for f in uploaded:
        with open(os.path.join(raw_dir, f.name), "wb") as out:
            out.write(f.getbuffer())
    st.sidebar.success(f"Saved {len(uploaded)} files to {raw_dir}.")
    if st.sidebar.button("Ingest uploaded docs"):
        with st.spinner("Ingesting..."):
            ingest_path(raw_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        st.sidebar.success("Ingestion complete.")

# Helper to render retrieved snippets (used in no-LLM mode)
def render_retrieved_snippets(docs, max_chars=800):
    st.markdown("### Top matches")
    for i, d in enumerate(docs, 1):
        meta = d.get("metadata", {})
        title = meta.get("doc_title", "Document")
        page = meta.get("page")
        slide = meta.get("slide_title")
        loc = f"p.{page}" if page is not None else ""
        if slide:
            loc = f"{loc} — {slide}" if loc else slide
        st.markdown(f"**[{i}] {title}** {loc}")
        txt = d.get("text", "")
        if len(txt) > max_chars:
            txt = txt[:max_chars] + "…"
        st.write(txt)

# Tabs
tab1, tab2, tab3 = st.tabs(["Ask", "Teach Me", "Quiz Me"])

with tab1:
    st.subheader("Ask a question")
    q = st.text_input("Your question", placeholder="e.g., What are common cost drivers in a value chain?")
    k = st.slider("Top-K retrieval", 2, 12, 6)
    if st.button("Get Answer", type="primary"):
        docs = retrieve_with_rerank(q, top_k=k)
        if not docs:
            st.warning("No relevant context found. Try ingesting more documents.")
        else:
            if LLM_PROVIDER == "none":
                render_retrieved_snippets(docs)
            else:
                ans = generate_answer(q, docs)
                st.markdown("### Answer")
                st.write(ans["text"])
                with st.expander("Citations"):
                    for i, c in enumerate(ans["citations"], 1):
                        title = c.get("doc_title", "Document")
                        page = c.get("page")
                        slide = c.get("slide_title")
                        loc = f"p.{page}" if page is not None else ""
                        if slide:
                            loc = f"{loc} — {slide}" if loc else slide
                        st.markdown(f"**[{i}] {title}** {loc}")

with tab2:
    st.subheader("Teach me a topic")
    topic = st.text_input("Topic", placeholder="e.g., Market entry basics")
    k2 = st.slider("Top-K retrieval (Teach)", 3, 15, 8)
    if st.button("Generate Summary"):
        docs = retrieve_with_rerank(topic, top_k=k2)
        if not docs:
            st.warning("No relevant context found.")
        else:
            if LLM_PROVIDER == "none":
                render_retrieved_snippets(docs)
            else:
                res = generate_summary(topic, docs)
                st.markdown("### Guided Summary")
                st.write(res["text"])
                with st.expander("Citations"):
                    for i, c in enumerate(res["citations"], 1):
                        title = c.get("doc_title", "Document")
                        page = c.get("page")
                        slide = c.get("slide_title")
                        loc = f"p.{page}" if page is not None else ""
                        if slide:
                            loc = f"{loc} — {slide}" if loc else slide
                        st.markdown(f"**[{i}] {title}** {loc}")

with tab3:
    st.subheader("Quiz me")
    topic_q = st.text_input("Topic for quiz", placeholder="e.g., Cost drivers")
    k3 = st.slider("Top-K retrieval (Quiz)", 3, 15, 8)
    if st.button("Make Quiz"):
        docs = retrieve_with_rerank(topic_q, top_k=k3)
        if not docs:
            st.warning("No relevant context found.")
        else:
            if LLM_PROVIDER == "none":
                st.info("Quizzes require an LLM. Showing top matches instead:")
                render_retrieved_snippets(docs)
            else:
                quiz = generate_quiz(topic_q, docs)
                st.markdown("### 5 Questions")
                for i, q_ in enumerate(quiz.get("questions", []), 1):
                    st.markdown(f"**{i}. {q_['question']}**")
                    for opt in q_["options"]:
                        st.markdown(f"- {opt}")
                    with st.expander("Answer & rationale"):
                        st.markdown(f"**Answer:** {q_['answer']}")
                        st.write(q_["rationale"])
