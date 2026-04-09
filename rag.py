"""
RAG query layer: retrieve relevant chunks → generate answer with an LLM.
Uses LCEL (LangChain Expression Language) — compatible with LangChain 0.3+.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say "I don't know based on the video."

Context:
{context}

Question: {question}

Answer:"""
)


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_qa_chain(vectorstore: FAISS):
    """Create an LCEL RAG chain from an existing FAISS vectorstore."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=os.environ["GROQ_API_KEY"],
    )

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return retriever, chain


def answer_question(retriever, chain, question: str) -> dict:
    """Run a question through the RAG chain and return answer + sources."""
    answer = chain.invoke(question)
    sources = retriever.invoke(question)
    return {
        "answer": answer,
        "sources": [doc.page_content[:200] for doc in sources],
    }
