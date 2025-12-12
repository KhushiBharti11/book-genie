# test_recommend.py — run this from your project folder to debug recommend_by_title
import joblib, pandas as pd, traceback, sys, os

print("cwd:", os.getcwd())

try:
    print("\nLoading bundle...")
    b = joblib.load("recommender_bundle.pkl")
    model = b["model"]
    book_pivot = b["book_pivot"]
    print("bundle keys loaded. book_pivot length:", len(book_pivot))
except Exception as e:
    print("Failed to load bundle:", e)
    traceback.print_exc()
    sys.exit(1)

try:
    print("\nLoading Books.csv...")
    books_df = pd.read_csv("Books.csv", low_memory=False)
    print("Books.csv loaded. Columns:", books_df.columns.tolist())
    # print first 5 rows of Book-Title if exists
    if "Book-Title" in books_df.columns:
        print("Sample titles (first 10):")
        print(books_df["Book-Title"].astype(str).head(10).tolist())
    else:
        # show first text-like column sample
        for c in books_df.columns:
            if books_df[c].dtype == object:
                print("Using sample column:", c)
                print(books_df[c].astype(str).head(10).tolist())
                break
except Exception as e:
    print("Failed to load Books.csv:", e)
    traceback.print_exc()
    sys.exit(1)

# change this test title to the one you used in Swagger (e.g. "Animal Farm")
TEST_TITLE = "Animal Farm"
N = 5

try:
    print(f"\nSearching for title like: {TEST_TITLE!r}")
    # pick title column
    if "Book-Title" in books_df.columns:
        title_col = "Book-Title"
    else:
        # fallback
        title_col = next((c for c in books_df.columns if books_df[c].dtype==object), books_df.columns[0])
    print("Using title column:", title_col)

    titles = books_df[title_col].astype(str)
    mask = titles.str.lower().str.contains(TEST_TITLE.lower(), na=False)
    matched = books_df[mask]
    print("count substring matches:", len(matched))

    if matched.empty:
        from difflib import get_close_matches
        close = get_close_matches(TEST_TITLE, titles.tolist(), n=3, cutoff=0.5)
        print("close matches suggestions:", close)
        if not close:
            print("No close matches — endpoint would return 404.")
            sys.exit(0)
        # use first close match
        candidate = close[0]
        idx = int(titles[titles == candidate].index[0])
    else:
        idx = int(matched.index[0])

    print("Chosen index for recommendation:", idx, " (0-based)")

    # sanity checks
    print("book_pivot shape:", getattr(book_pivot, "shape", "unknown"))
    if idx < 0 or idx >= len(book_pivot):
        print("ERROR: chosen index is out of bounds for book_pivot.")
        sys.exit(1)

    print("\nCalling model.kneighbors for index", idx)
    dists, neighs = model.kneighbors(book_pivot.iloc[idx, :].values.reshape(1, -1), n_neighbors=N+1)
    print("kneighbors returned distances length:", len(dists[0]), "neighbors:", neighs[0].tolist())

    rec_idxs = [int(i) for i in neighs[0] if int(i) != idx][:N]
    print("Final recommended indices:", rec_idxs)

    # map to metadata safely
    def idx_to_meta(i):
        if 0 <= i < len(books_df):
            row = books_df.iloc[i]
            return {
                "index": int(i),
                "title": row.get("Book-Title", ""),
                "author": row.get("Book-Author", ""),
                "image": row.get("Image-URL-L", None)
            }
        return {"index": int(i)}

    print("\nSample metadata for recs:")
    print([idx_to_meta(i) for i in rec_idxs])

    print("\nALL OK — recommend_by_title logic works locally.")
except Exception as e:
    print("EXCEPTION during test run:", e)
    traceback.print_exc()
    sys.exit(1)
