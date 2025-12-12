# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from difflib import get_close_matches
from typing import List, Dict, Optional

# --------------------------
# APP + CORS
# --------------------------
app = FastAPI(title="Book Recommender API")

# Allow local React dev server to call this API
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# LOAD MODEL + DATA
# --------------------------
bundle = joblib.load("recommender_bundle.pkl")
model = bundle["model"]
book_pivot = bundle["book_pivot"]

# load metadata csv (change filename if needed)
try:
    books_df = pd.read_csv("Books.csv", low_memory=False)
except Exception:
    books_df = pd.DataFrame()

# --------------------------
# HELPERS
# --------------------------
def idxs_to_metadata(indices: List[int]) -> List[Dict]:
    results = []
    for idx in indices:
        if 0 <= idx < len(books_df):
            row = books_df.iloc[idx]
            # use exact column names from your CSV
            title = row.get("Book-Title", "")
            author = row.get("Book-Author", "")
            image = row.get("Image-URL-L", None) if "Image-URL-L" in books_df.columns else None
            results.append({
                "index": int(idx),
                "title": str(title),
                "author": str(author),
                "image_url": image
            })
        else:
            results.append({"index": int(idx)})
    return results

def find_title_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.shape[0] == 0:
        return None
    # prefer the known column
    if "Book-Title" in df.columns:
        return "Book-Title"
    # other fallbacks
    candidates = [c for c in df.columns if c.lower() in ("title", "book_title", "bookname", "name", "book")]
    if candidates:
        return candidates[0]
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return df.columns[0] if len(df.columns) > 0 else None

# --------------------------
# REQUEST MODELS
# --------------------------
class IndexQuery(BaseModel):
    index: int
    n: int = 5

class TitleQuery(BaseModel):
    title: str
    n: int = 5

# --------------------------
# ENDPOINT: recommend by index
# --------------------------
@app.post("/recommend")
def recommend_books(q: IndexQuery):
    idx = q.index
    if idx < 0 or idx >= len(book_pivot):
        raise HTTPException(status_code=400, detail="Index out of range")
    distances, neighbors = model.kneighbors(
        book_pivot.iloc[idx, :].values.reshape(1, -1),
        n_neighbors=q.n + 1
    )
    rec_idxs = [int(i) for i in neighbors[0] if int(i) != idx][: q.n]
    return {
        "query_index": idx,
        "recommendations": idxs_to_metadata(rec_idxs),
        "distances": distances[0].tolist()
    }

# --------------------------
# ENDPOINT: recommend by title (robust)
# --------------------------
@app.post("/recommend_by_title")
def recommend_by_title(q: TitleQuery):
    try:
        if books_df is None or books_df.shape[0] == 0:
            raise HTTPException(status_code=500, detail="Books.csv is missing or empty on the server.")

        title_input = q.title.strip()
        if not title_input:
            raise HTTPException(status_code=400, detail="Empty title provided.")

        title_col = find_title_column(books_df)
        if title_col is None:
            raise HTTPException(status_code=500, detail="Could not determine title column from Books.csv.")

        titles_series = books_df[title_col].astype(str)

        # substring (case-insensitive)
        mask = titles_series.str.lower().str.contains(title_input.lower(), na=False)
        matches = books_df[mask]

        # fuzzy fallback
        if matches.empty:
            close = get_close_matches(title_input, titles_series.tolist(), n=1, cutoff=0.55)
            if not close:
                raise HTTPException(status_code=404, detail="No title match found.")
            matched_title = close[0]
            matches = books_df[titles_series == matched_title]

        idx = int(matches.index[0])

        distances, neighbors = model.kneighbors(
            book_pivot.iloc[idx, :].values.reshape(1, -1),
            n_neighbors=q.n + 1
        )
        rec_idxs = [int(i) for i in neighbors[0] if int(i) != idx][: q.n]

        return {
            "query_title": str(books_df.iloc[idx].get(title_col, "")),
            "query_index": idx,
            "recommendations": idxs_to_metadata(rec_idxs),
            "distances": distances[0].tolist()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
@app.get("/suggest_titles")
def suggest_titles(q: str):
    if not q:
        return []
    
    title_col = "Book-Title"
    titles = books_df[title_col].astype(str)

    matches = titles[titles.str.lower().str.contains(q.lower())].head(10)

    return matches.to_list()
# --------------------------
# ENDPOINT: Get all books
# --------------------------
@app.get("/all_books")
def all_books(limit: int = 500):
    """
    Returns up to 'limit' books with index, title, author, and image.
    """
    if books_df is None or books_df.empty:
        raise HTTPException(status_code=500, detail="Books.csv unavailable")

    title_col = find_title_column(books_df)

    data = []
    for idx, row in books_df.head(limit).iterrows():
        data.append({
            "index": idx,
            "title": str(row.get(title_col, "")),
            "author": str(row.get("Book-Author", "")),
            "image_url": row.get("Image-URL-L", None)
        })

    return {"count": len(data), "books": data}
