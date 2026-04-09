"""
LangGraph pipeline: Transcript → Clean → Chunk → Store in FAISS
Runs once per video URL.
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from utils.transcript import extract_video_id, get_transcript
from utils.cleaner import clean_text
from utils.chunker import chunk_text


EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


class PipelineState(TypedDict):
    url: str
    video_id: Optional[str]
    raw_text: Optional[str]
    clean_text: Optional[str]
    chunks: Optional[list[str]]
    vectorstore: Optional[object]
    error: Optional[str]


# ── Nodes ──────────────────────────────────────────────────────────────────────

def node_get_transcript(state: PipelineState) -> PipelineState:
    try:
        video_id = extract_video_id(state["url"])
        raw = get_transcript(video_id)
        return {**state, "video_id": video_id, "raw_text": raw, "error": None}
    except Exception as e:
        return {**state, "error": str(e)}


def node_clean_text(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state
    cleaned = clean_text(state["raw_text"])
    return {**state, "clean_text": cleaned}


def node_chunk_text(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state
    chunks = chunk_text(state["clean_text"])
    return {**state, "chunks": chunks}


def node_store_vectors(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state
    try:
        vs = FAISS.from_texts(state["chunks"], EMBEDDINGS)
        return {**state, "vectorstore": vs}
    except Exception as e:
        return {**state, "error": str(e)}


# ── Graph ──────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("get_transcript", node_get_transcript)
    graph.add_node("clean_text", node_clean_text)
    graph.add_node("chunk_text", node_chunk_text)
    graph.add_node("store_vectors", node_store_vectors)

    graph.set_entry_point("get_transcript")
    graph.add_edge("get_transcript", "clean_text")
    graph.add_edge("clean_text", "chunk_text")
    graph.add_edge("chunk_text", "store_vectors")
    graph.add_edge("store_vectors", END)

    return graph.compile()


def process_video(url: str) -> PipelineState:
    """Run the full ingestion pipeline for a YouTube URL."""
    pipeline = build_graph()
    initial_state: PipelineState = {
        "url": url,
        "video_id": None,
        "raw_text": None,
        "clean_text": None,
        "chunks": None,
        "vectorstore": None,
        "error": None,
    }
    return pipeline.invoke(initial_state)
