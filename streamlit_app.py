# streamlit_app.py
import os
import io
import re
import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import streamlit as st

# ML libs
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader

# transformer pipeline (local generation)
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -----------------------
# Config & folders
# -----------------------
DATA_DIR = Path("uploaded_docs")
INDEX_DIR = Path("rag_index")
DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)
META_PATH = INDEX_DIR / "metadata.pkl"
INDEX_PATH = INDEX_DIR / "faiss.index"

st.set_page_config(page_title="RAG App — Upload PDF & QA", layout="wide")
st.title("📚 RAG QA — Upload PDF / Markdown and Ask Questions")

# -----------------------
# Utilities: chunking, parsing
# -----------------------
def split_into_sentences(text: str):
    return [s.strip() for s in re.split(r'(?<=[\.\?\!\n])\s+', text.strip()) if s.strip()]

def chunk_text_by_sentences(text: str, max_chars: int = 1000, overlap: int = 200):
    sents = split_into_sentences(text)
    chunks, cur, cur_len = [], [], 0
    for sent in sents:
        if cur and (cur_len + len(sent) > max_chars):
            chunks.append(" ".join(cur).strip())
            if overlap > 0:
                ov_cur, ov_len = [], 0
                for s in reversed(cur):
                    ov_cur.insert(0, s)
                    ov_len += len(s)
                    if ov_len >= overlap:
                        break
                cur = ov_cur
                cur_len = sum(len(x) for x in cur)
            else:
                cur = []
                cur_len = 0
        cur.append(sent)
        cur_len += len(sent)
    if cur:
        chunks.append(" ".join(cur).strip())
    return chunks

def parse_pdf_bytes(b: bytes):
    reader = PdfReader(io.BytesIO(b))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append({"page_num": i + 1, "text": text})
    return pages

def parse_markdown_bytes(b: bytes):
    text = b.decode("utf-8", errors="ignore")
    return [{"page_num": None, "text": text}]

def ingest_bytes(b: bytes, filename: str, chunk_chars=1000, overlap=200):
    filename = Path(filename).name
    ext = Path(filename).suffix.lower()
    docs = []
    if ext == ".pdf":
        pages = parse_pdf_bytes(b)
        for p in pages:
            text = p["text"].strip()
            if not text:
                continue
            chunks = chunk_text_by_sentences(text, max_chars=chunk_chars, overlap=overlap)
            for i, c in enumerate(chunks):
                docs.append({
                    "text": c,
                    "source": filename,
                    "page": p["page_num"],
                    "chunk_id": f"{filename}::p{p['page_num']}::c{i}"
                })
    elif ext in [".md", ".markdown", ".txt"]:
        pages = parse_markdown_bytes(b)
        for p in pages:
            chunks = chunk_text_by_sentences(p["text"], max_chars=chunk_chars, overlap=overlap)
            for i, c in enumerate(chunks):
                docs.append({
                    "text": c,
                    "source": filename,
                    "page": None,
                    "chunk_id": f"{filename}::c{i}"
                })
    else:
        raise ValueError("Unsupported file type: " + ext)
    return docs

# -----------------------
# Caching: models
# -----------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

@st.cache_resource(show_spinner=False)
def load_local_generator(model_name="google/flan-t5-base"):
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text2text-generation", model=model_name, device=device)
    return pipe

# -----------------------
# Indexing helpers (FAISS)
# -----------------------
def build_faiss_index(docs: List[Dict[str, Any]]):
    texts = [d["text"] for d in docs]
    embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.asarray(embeddings).astype("float32"))
    # persist
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(docs, f)
    return index

def load_faiss_index():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        return None, None
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def add_docs_to_index(new_docs: List[Dict[str, Any]]):
    index, metadata = load_faiss_index()
    if index is None:
        return build_faiss_index(new_docs)
    new_texts = [d["text"] for d in new_docs]
    new_embs = embed_model.encode(new_texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    index.add(new_embs)
    metadata.extend(new_docs)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)
    return index

# -----------------------
# Retrieval + prompt builder
# -----------------------
def retrieve(query: str, top_k: int = 5):
    index, metadata = load_faiss_index()
    if index is None:
        return []
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        md = metadata[idx]
        results.append({"score": float(score), "metadata": md, "text": md["text"]})
    return results

def build_prompt_with_context(question: str, retrieved: List[Dict[str, Any]]):
    ctx_parts = []
    for i, r in enumerate(retrieved, start=1):
        md = r["metadata"]
        header = f"[{i}] {md['source']} | page:{md['page']} | chunk:{md['chunk_id']}"
        ctx_parts.append(header + "\n" + r["text"])
    context = "\n\n---\n\n".join(ctx_parts)
    prompt = f"""You are an accurate assistant. Use ONLY the information in the CONTEXT to answer the QUESTION.
For each claim that comes from the context, include reference markers like [1], [2], ... that map to the sources below. If the answer cannot be found in the context, say "I don't know based on the provided document."

CONTEXT:
{context}

QUESTION: {question}

ANSWER (concise, with sources):"""
    return prompt

# -----------------------
# Generation wrappers
# -----------------------
def generate_with_local(prompt: str, local_model):
    out = local_model(prompt, max_new_tokens=256, do_sample=False)
    # pipeline returns list of dicts
    text = out[0].get("generated_text") or out[0].get("summary_text") or str(out[0])
    return text

def generate_with_hf_inference(prompt: str, hf_model="google/flan-t5-base"):
    # uses InferenceClient if available & user provided HF_TOKEN
    from huggingface_hub import InferenceClient
    HF_TOKEN = os.getenv("HF_TOKEN") or (st.secrets.get("HF_TOKEN") if "HF_TOKEN" in st.secrets else None)
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not found. Set HF_TOKEN in environment or Streamlit secrets.")
    client = InferenceClient(model=hf_model, token=HF_TOKEN)
    return client.text_generation(prompt, max_new_tokens=256)

# -----------------------
# UI: sidebar
# -----------------------
with st.sidebar:
    st.header("Settings")
    chunk_chars = st.number_input("Chunk size (chars)", min_value=200, max_value=5000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=2000, value=200, step=50)
    top_k = st.slider("Retriever: top_k", 1, 10, 5)
    generation_backend = st.radio("Generation backend", ("Local model (transformers)", "Hugging Face Inference API"))
    local_model_name = st.selectbox("Local model (if using local backend)", ("google/flan-t5-small", "google/flan-t5-base"))
    hf_model_name = st.text_input("HF model (if using HF API)", value="google/flan-t5-base")
    st.markdown("---")
    st.caption("Tip: set HF_TOKEN in Streamlit secrets for HF API usage.")

# -----------------------
# UI: Upload / ingest
# -----------------------
uploaded_files = st.file_uploader("Upload PDF / Markdown (multiple allowed)", type=["pdf", "md", "markdown", "txt"], accept_multiple_files=True)

if uploaded_files:
    st.info("Click 'Ingest & Index' to process uploaded file(s) and build/update the vector index.")
    if st.button("Ingest & Index"):
        embed_load_spinner = st.empty()
        embed_load_spinner.info("Loading embedding model...")
        # load embed model once
        global embed_model
        embed_model = load_embedding_model()
        embed_load_spinner.success("Embedding model loaded.")
        progress = st.progress(0)
        all_new_docs = []
        for i, up in enumerate(uploaded_files):
            b = up.read()
            # save original file
            dest = DATA_DIR / up.name
            with open(dest, "wb") as f:
                f.write(b)
            # ingest with current UI chunk params
            docs = ingest_bytes(b, up.name, chunk_chars, chunk_overlap)
            all_new_docs.extend(docs)
            progress.progress(int(((i+1) / len(uploaded_files)) * 50))
            st.write(f"Extracted {len(docs)} chunks from {up.name}")
        # add to index (create or append)
        progress.progress(60)
        index, metadata = load_faiss_index()
        if index is None:
            st.write("Creating new FAISS index...")
            build_faiss_index(all_new_docs)
            st.success(f"Built index with {len(all_new_docs)} chunks.")
        else:
            st.write("Adding chunks to existing index...")
            add_docs_to_index(all_new_docs)
            st.success("Index updated.")
        progress.progress(100)
        st.balloons()

# -----------------------
# UI: Show index info
# -----------------------
index, metadata = load_faiss_index()
if index is not None and metadata:
    st.sidebar.success(f"Index ready — {index.ntotal} vectors stored.")
else:
    st.sidebar.warning("No index found. Upload a document to start.")

# -----------------------
# UI: Query
# -----------------------
st.markdown("## Ask a question about the uploaded document")
question = st.text_input("Type your question here:")

col1, col2 = st.columns([3,1])
with col2:
    run_btn = st.button("Answer")

if run_btn and question.strip():
    if index is None:
        st.error("Index not found. Upload & index documents first.")
    else:
        with st.spinner("Retrieving relevant chunks..."):
            # ensure embed model loaded
            embed_model = load_embedding_model()
            retrieved = retrieve(question, top_k=top_k)
        if not retrieved:
            st.warning("No relevant context found.")
        else:
            st.write("### Retrieved passages (for transparency)")
            for r in retrieved:
                md = r["metadata"]
                st.write(f"**{md['source']}** — page {md['page']} — `{md['chunk_id']}` — score={r['score']:.4f}")
                st.write(r["text"][:800] + ("..." if len(r["text"]) > 800 else ""))
                st.markdown("---")

            prompt = build_prompt_with_context(question, retrieved)
            st.write("### Prompt sent to LLM (truncated)")
            st.code(prompt[:4000] + ("\n\n... (truncated)" if len(prompt) > 4000 else ""), language="text")

            # generation
            if generation_backend.startswith("Local"):
                # load local generator
                with st.spinner(f"Loading local model {local_model_name} (this may take time)..."):
                    local_gen = load_local_generator(local_model_name)
                with st.spinner("Generating answer..."):
                    answer_text = generate_with_local(prompt, local_gen)
            else:
                # HF inference API
                try:
                    with st.spinner("Generating via Hugging Face Inference API..."):
                        answer_text = generate_with_hf_inference(prompt, hf_model=hf_model_name)
                except Exception as e:
                    st.error("HF Inference failed: " + str(e))
                    answer_text = "Error: generation failed."

            st.markdown("## ✅ Answer")
            st.write(answer_text)

            st.markdown("### Sources used (retrieved chunks)")
            for i, r in enumerate(retrieved, start=1):
                md = r["metadata"]
                st.write(f"{i}. **{md['source']}** | page: {md['page']} | chunk_id: `{md['chunk_id']}` | score: {r['score']:.4f}")
                # optionally show small excerpt
                st.write(">" + (md["text"][:300].replace("\n", " ") + ("..." if len(md["text"]) > 300 else "")))

# -----------------------
# Utilities: download index / metadata
# -----------------------
st.markdown("---")
st.markdown("### Index & metadata")
if INDEX_PATH.exists() and META_PATH.exists():
    if st.button("Download metadata.pkl"):
        with open(META_PATH, "rb") as f:
            st.download_button(label="Download metadata.pkl", data=f.read(), file_name="metadata.pkl")
    if st.button("Download faiss.index"):
        # write to memory then download
        faiss.write_index(faiss.read_index(str(INDEX_PATH)), "tmp.index")
        with open("tmp.index", "rb") as f:
            st.download_button(label="Download faiss.index", data=f.read(), file_name="faiss.index")
else:
    st.info("Index files not present yet.")
