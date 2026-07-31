"""Run once (locally or as a Render pre-deploy step) to populate the
vector store from backend/data/docs. Re-run any time you add new docs."""
from app.vectorstore import ingest_folder

if __name__ == "__main__":
    n = ingest_folder("./data/docs")
    print(f"Ingested {n} chunks.")
