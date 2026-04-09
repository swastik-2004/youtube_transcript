"""
Streamlit UI — YouTube RAG App
"""

import streamlit as st
from dotenv import load_dotenv
load_dotenv()  # loads .env before any other import reads os.environ

from graph import process_video
from rag import build_qa_chain, answer_question

st.set_page_config(page_title="YouTube RAG", page_icon="🎬", layout="centered")
st.title("🎬 YouTube → RAG Q&A")

# ── Session state ──────────────────────────────────────────────────────────────
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "video_url" not in st.session_state:
    st.session_state.video_url = ""

# ── Sidebar: video ingestion ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📥 Load a Video")
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

    if st.button("Process Video", type="primary", disabled=not url):
        if url != st.session_state.video_url:
            with st.spinner("Fetching transcript and building index…"):
                result = process_video(url)

            if result["error"]:
                st.error(f"Error: {result['error']}")
            else:
                retriever, chain = build_qa_chain(result["vectorstore"])
                st.session_state.retriever = retriever
                st.session_state.chain = chain
                st.session_state.video_url = url
                chunk_count = len(result["chunks"])
                st.success(f"Ready! Indexed {chunk_count} chunks.")
        else:
            st.info("This video is already loaded.")

    if st.session_state.video_url:
        st.caption(f"Loaded: {st.session_state.video_url}")

# ── Main: Q&A ─────────────────────────────────────────────────────────────────
if st.session_state.chain is None:
    st.info("Paste a YouTube URL in the sidebar and click **Process Video** to begin.")
else:
    st.subheader("Ask a question about the video")

    question = st.text_input("Your question", placeholder="What is the main topic discussed?")

    if st.button("Get Answer", type="primary", disabled=not question):
        with st.spinner("Thinking…"):
            output = answer_question(st.session_state.retriever, st.session_state.chain, question)

        st.markdown("### Answer")
        st.write(output["answer"])

        with st.expander("Retrieved context chunks"):
            for i, src in enumerate(output["sources"], 1):
                st.markdown(f"**Chunk {i}:** {src}…")
