# 🎬 YouTube → RAG Q&A App

A local RAG (Retrieval-Augmented Generation) app that lets you ask questions about any YouTube video using its transcript.



---

## 🚀 How It Works

```
YouTube URL
    ↓
Transcript Extraction (youtube-transcript-api)
    ↓
Text Cleaning + Chunking (LangChain)
    ↓
Embeddings + Vector Store (HuggingFace + FAISS)
    ↓
User Question → RAG → Answer (Groq LLM)
```

---

## 🧱 Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Pipeline | LangGraph |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector DB | FAISS (in-memory) |
| LLM | Groq `llama-3.1-8b-instant` (free) |
| Transcript | youtube-transcript-api |

---

## ⚙️ Setup

**1. Clone the repo**
```bash
git clone https://github.com/swastik-2004/youtube-rag-app.git
cd youtube-rag-app
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up your API key**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

**4. (Optional) Add cookies to bypass YouTube IP blocks**

Export your YouTube browser cookies as `cookies.txt` using the
[Get cookies.txt LOCALLY](https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) extension and place it in the project root.

**5. Run the app**
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
youtube-rag-app/
│
├── app.py              # Streamlit UI
├── graph.py            # LangGraph ingestion pipeline
├── rag.py              # RAG chain (retriever + LLM)
│
├── utils/
│   ├── transcript.py   # YouTube transcript fetcher
│   ├── cleaner.py      # Text cleaning
│   └── chunker.py      # Text splitting
│
├── .env.example
├── .gitignore
└── requirements.txt
```

---

## 💡 Features

- Paste any YouTube URL and instantly index its transcript
- Ask free-form questions — answers are grounded in the video content
- Expandable source chunks show exactly what context the LLM used
- Same video won't be re-processed if already loaded
