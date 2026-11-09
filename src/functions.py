"""Your reusable functions live here.

The rule from the brief: Python logic goes in `.py` files, and the notebooks
hold the narrative. The app imports the same functions, so the text it receives
goes through exactly the same cleaning as your training data.

To use this module from a notebook in `notebooks/`:

    import sys
    sys.path.append("..")
    from src.functions import *

and from the app in `app/`, the same two lines with `sys.path.append(".")` when
you run `streamlit run app/streamlit_app.py` from the repo root.

The stubs below only show the shape. Rename them, change the arguments, delete
the ones for the other track, write your own.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"

RANDOM_STATE = 42
LLM_MODEL = "gpt-5.4-nano"                                    # LLM track: one place to change the model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# Both tracks

def load_reviews():
    """Read the raw reviews (and, for Amazon, join the product metadata you need)."""
    pass




def clean_text(text):
    """Clean text extracted from the forest entomology PDF."""
    import re
    import unicodedata

    if not isinstance(text, str):
        return ""

    # 1. Repair words broken across lines
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # 2. Replace PDF extraction artifacts BEFORE Unicode normalization
    text = re.sub(r"[`´ˆ]", " ", text)

    # 3. Normalize ligatures such as ﬁ -> fi
    text = unicodedata.normalize("NFKC", text)

    # 4. Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# LLM track

def build_index(documents):
    """Embed the chunks and store them, with their metadata, in a Chroma collection."""
    pass


def answer_question(question, collection, k=5):
    """Retrieve, prompt and answer; return the answer, the cited ids, the retrieved ids and the tokens."""
    pass


def evaluate(questions, answer_fn):
    """Run the evaluation set and return one row per question: hit, cites, refused, cost."""
    pass

def chunk_document(pages, size=150, overlap=30):
    """Split PDF pages into overlapping chunks with page metadata."""

    chunks = []
    step = size - overlap

    if step <= 0:
        raise ValueError("Overlap must be smaller than chunk size.")

    for _, page in pages.iterrows():
        words = page["clean_text"].split()
        pdf_page = int(page["pdf_page"])

        for start in range(0, len(words), step):
            chunk_words = words[start:start + size]

            if not chunk_words:
                continue

            chunks.append({
                "id": f"page_{pdf_page}_chunk_{start}",
                "text": " ".join(chunk_words),
                "pdf_page": pdf_page
            })

    return chunks

def search_chroma(question, model, collection, k=5, pdf_page=None):
    """Retrieve relevant textbook chunks from Chroma."""

    # Convert the question into an embedding
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True)

    # Apply an optional PDF page filter
    filters = {"pdf_page": pdf_page} if pdf_page is not None else None

    # Search the database
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=k,
        where=filters,
        include=["documents", "metadatas", "distances"]
    )
    return results


def build_prompt(question, results):
    """Build a grounded prompt using retrieved textbook passages."""

    # Combine the retrieved passages with their source information
    sources = []

    for i in range(len(results["ids"][0])):
        chunk_id = results["ids"][0][i]
        page = results["metadatas"][0][i]["pdf_page"]
        text = results["documents"][0][i]

        sources.append(
            f"Source ID: {chunk_id}\n"
            f"PDF page: {page}\n"
            f"Text: {text}"
        )

    context = "\n\n".join(sources)

    # Define the rules for the assistant
    prompt = f"""
You are the polite and helpful Forest Pest Assistant, helping garden
and small forest owners understand possible insect-related tree problems.

Answer questions using ONLY the textbook sources provided below.

Rules:
- Use only information supported by the provided sources.
- Cite supporting evidence using PDF page numbers, for example [PDF p. 326].
- Do not invent facts or references.
- Do not identify an exact insect species from insufficient symptoms.
- Do not recommend specific pesticides or dosages.
- Do not predict whether a tree will die without sufficient evidence.
- Keep answers clear, polite, and understandable for non-experts.

Clarifying questions:
- If important information is missing, ask a relevant clarifying question.
- Ask only one clarifying question at a time.
- Ask no more than five clarifying questions in total.
- Do not suggest that a diagnosis is certain when the evidence is insufficient.
- If the information is still insufficient after five clarifying questions, say:
  "Unfortunately, I do not have enough information to give you a reliable answer."

Insufficient textbook evidence:
- If the retrieved textbook sources do not support an answer, say:
  "I don't know based on the available textbook evidence."
- Do not ask clarifying questions when the missing information is absent
  from the textbook rather than from the user's description.

Textbook sources:
{context}

Question:
{question}

Answer:
"""

    return prompt

def generate_answer(prompt, client, model_name="gpt-6-luna"):
    """Generate an answer using the OpenAI API."""

    # Send the grounded prompt to the language model
    response = client.responses.create(
        model=model_name,
        input=prompt
    )

    # Return the generated text
    return response.output_text