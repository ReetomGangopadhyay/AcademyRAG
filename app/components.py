import streamlit as st

def answer_card(answer_text: str, citations: list):
    with st.container(border=True):
        st.markdown("### Answer")
        st.write(answer_text)
        if citations:
            with st.expander("Citations"):
                for i, c in enumerate(citations, 1):
                    title = c.get("doc_title", "Document")
                    page = c.get("page")
                    slide = c.get("slide_title")
                    loc = f"p.{page}" if page is not None else ""
                    if slide:
                        loc = f"{loc} — {slide}" if loc else slide
                    st.markdown(f"**[{i}] {title}** {loc}")
