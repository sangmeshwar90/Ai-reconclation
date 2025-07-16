from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os, fitz, pickle, numpy as np, atexit
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# === App and CORS ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Config ===
DOCUMENTS_FOLDER = "documents"
CHUNK_CACHE_FILE = "chunked_embeddings.pkl"
ALT_CACHE_FILE = "document_embeddings.pkl"
CHUNK_SIZE = 2
TOP_K = 3
MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_ENABLED = True
GEMINI_API_KEY = "AIzaSyBPbmhB_3Nxnkgn9RrfqfPtgluoqRmKWUM"
THREAD_WORKERS = 4

model = SentenceTransformer(MODEL_NAME)
genai.configure(api_key=GEMINI_API_KEY)

# === Logging ===
def log(msg):
    print(f"[LOG] {msg}")

# === Extract Chunks ===
def extract_chunks(path, chunk_size=CHUNK_SIZE):
    try:
        doc = fitz.open(path)
        chunks = []
        for i in range(0, len(doc), chunk_size):
            chunk_text = " ".join([doc[j].get_text() for j in range(i, min(i + chunk_size, len(doc)))])
            if chunk_text.strip():
                chunks.append((i, chunk_text))
        log(f"✅ Extracted {len(chunks)} chunks from {path}")
        return chunks
    except Exception as e:
        log(f"❌ Failed to extract chunks: {e}")
        return []

# === Cosine Similarity ===
def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# === Process One PDF ===
def process_pdf_file(path):
    try:
        filename = os.path.basename(path)
        log(f"🔍 Processing: {filename}")
        chunks = extract_chunks(path)
        vectors = []
        for start_page, text in chunks:
            try:
                emb = model.encode(text)
                vectors.append({"start_page": start_page, "embedding": emb})
            except Exception as e:
                log(f"❌ Encoding error in {filename} page {start_page}: {e}")
        return filename, vectors
    except Exception as e:
        log(f"❌ Failed processing {path}: {e}")
        return None, []

# === Refresh Embeddings (Multithreaded) ===
def refresh_embeddings():
    log("🔁 Refreshing embeddings using ThreadPoolExecutor...")
    chunked_embeddings = {}
    pdf_paths = [
        os.path.join(DOCUMENTS_FOLDER, file)
        for file in os.listdir(DOCUMENTS_FOLDER)
        if file.endswith(".pdf")
    ]

    with ThreadPoolExecutor(max_workers=THREAD_WORKERS) as executor:
        futures = [executor.submit(process_pdf_file, path) for path in pdf_paths]
        for future in as_completed(futures):
            filename, vectors = future.result()
            if filename and vectors:
                chunked_embeddings[filename] = vectors
                log(f"✅ Finished {filename} with {len(vectors)} embeddings.")

    with open(CHUNK_CACHE_FILE, "wb") as f:
        pickle.dump(chunked_embeddings, f)
        log(f"💾 Saved embeddings to {CHUNK_CACHE_FILE}")
    return chunked_embeddings

# === Load or Create Cache ===
def load_or_create_chunk_embeddings():
    if os.path.exists(CHUNK_CACHE_FILE):
        try:
            with open(CHUNK_CACHE_FILE, "rb") as f:
                log("📂 Loaded embeddings from cache.")
                return pickle.load(f)
        except Exception as e:
            log(f"❌ Failed to load cache: {e}")
    return refresh_embeddings()

# === Startup Embedding Load ===
stored_chunks = load_or_create_chunk_embeddings()

# === Gemini Classification ===
def classify_with_gemini(text):
    prompt = f"""
This is a document. Identify what kind of document it is (e.g., invoice, contract, payslip, bank statement, etc.)
You can only reply with 1–3 words.

Document:
{text[:1500]}
"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        log("✨ Gemini classification complete.")
        return response.text.strip()
    except Exception as e:
        log(f"❌ Gemini failed: {e}")
        return "Unknown"

# === Match Document ===
@app.post("/match-document/")
async def match_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return JSONResponse(status_code=400, content={"error": "Only PDF files are supported."})

    try:
        contents = await file.read()
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(contents)

        log(f"📥 Uploaded file received: {file.filename}")
        chunks = extract_chunks("temp_uploaded.pdf")
        os.remove("temp_uploaded.pdf")

        if not chunks:
            return JSONResponse(status_code=400, content={"error": "No readable text found in PDF."})

        results = []
        for _, query_text in chunks:
            query_emb = model.encode(query_text)
            for filename, file_chunks in stored_chunks.items():
                for item in file_chunks:
                    score = cosine_sim(query_emb, item["embedding"])
                    results.append({
                        "file": filename,
                        "score": score,
                        "start_page": item["start_page"]
                    })

        if not results:
            return {"message": "No match found."}

        top = sorted(results, key=lambda x: -x["score"])[:TOP_K]
        best_score = top[0]['score']

        response = {
            "top_matches": [
                {
                    "file": r["file"],
                    "similarity": round(r["score"], 3),
                    "matched_start_page": r["start_page"]
                } for r in top
            ]
        }

        if GEMINI_ENABLED and best_score < 0.65:
            sample_text = chunks[0][1]
            response["gemini_guess"] = classify_with_gemini(sample_text)

        return response

    except Exception as e:
        log(f"❌ Error in /match-document: {e}")
        return JSONResponse(status_code=500, content={"error": f"Internal error: {str(e)}"})

# === Refresh Cache ===
@app.get("/refresh-cache/")
def refresh_cache():
    global stored_chunks
    stored_chunks = refresh_embeddings()
    return {"message": "Document cache refreshed."}

# === Clean Up on Exit ===
def cleanup_files():
    for f in [CHUNK_CACHE_FILE, ALT_CACHE_FILE]:
        try:
            if os.path.exists(f):
                os.remove(f)
                log(f"🧹 Deleted file on exit: {f}")
        except Exception as e:
            log(f"❌ Failed to delete {f}: {e}")

atexit.register(cleanup_files)

# === Ping ===
@app.get("/")
def ping():
    return {"status": "ok"}
