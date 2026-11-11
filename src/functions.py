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

# geting the right page from the book
def get_book_page(text):
    """Extract the printed page number from a textbook header."""
    import re

    header = text.strip().splitlines()[0]

    # Chapter-opening pages have no normal page header
    if header.startswith("Chapter"):
        return None

    # Allow spaces between an author's initial and the period
    author_header = re.match(r"^(\d+)\s+[A-Z]\s*\.", header)

    if author_header:
        return int(author_header.group(1))

    # Chapter headers: "9 Foliage Feeders 267Table ..."
    chapter_header = re.match(
        r"^\d+\s+.+?\s+(\d+)(?=Table|\s|$)",
        header
    )

    if chapter_header:
        return int(chapter_header.group(1))

    return None

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


def build_prompt(question, results, history=None):
    """Build a grounded prompt using retrieved textbook sources."""

    sources = []

    # Collect the retrieved text and its citation information
    for i in range(len(results["ids"][0])):
        source_id = results["ids"][0][i]
        text = results["documents"][0][i]
        metadata = results["metadatas"][0][i]

        # Prefer the printed textbook page when available
        pdf_page = metadata["pdf_page"]
        book_page = metadata.get("book_page")

        if book_page is not None:
            citation = f"Book p. {book_page}"
        else:
            citation = f"PDF p. {pdf_page}"

        source = (
            f"Source ID: {source_id}\n"
            f"Citation: [{citation}]\n"
            f"Text: {text}"
        )

        sources.append(source)

    # Combine all retrieved sources into one context
    context = "\n\n".join(sources)

    # formating previous messages for the assistant to follow
    conversation = []
    if history:
        for message in history:
            role = message["role"]
            content = message["content"]
            conversation.append(f"{role}:{content}")

    conversation_text = "\n".join(conversation)

    # Give the assistant its instructions and textbook evidence
    prompt = f"""
You are the polite and helpful Tree Health Investigator, helping garden
and small forest owners understand possible insect-related tree problems.

Answer questions using ONLY the textbook sources provided below.

Rules:
- Base your answers only on the provided textbook sources and the user's observations. Do not invent facts, references, or page numbers.
- Cite supporting textbook evidence using the exact citation labels provided. Prefer [Book p. X] when available; otherwise use [PDF p. X].
- Use the conversation history to remember observations, previous conclusions, and questions already answered. Previous assistant responses are not independent textbook evidence.
- Focus on the user's newest information and build on earlier findings without repeating explanations unnecessarily.
- Identify the most plausible insect group or species when the observations and textbook evidence support it. Clearly distinguish a tentative identification from a confirmed one.
- Explain uncertainty when relevant, but do not withhold a useful, evidence-supported hypothesis simply because an exact identification cannot be confirmed.
- Do not recommend specific pesticides or dosages, or predict tree death without sufficient evidence.
- Keep answers clear, concise, polite, and understandable for non-experts.

Clarifying questions:
- When a useful observation could help narrow down the possible insect
  group or species, ask one specific clarification question, even if
  you can already offer an initial hypothesis.
- Ask at most one question per response and no more than five during an investigation.
- Do not repeat questions already answered, including indirectly.
  If the user says they do not know or cannot check something,
  treat that information as unavailable and do not ask for it again
  in a different form.
  - Interpret short replies in the context of the preceding question.
- Provide the best-supported hypothesis at each stage of the investigation.
  Continue asking questions only when a new, realistically observable
  detail could meaningfully narrow the identification.
  Stop when no useful question remains or when the user cannot provide
  the information needed to narrow it further.
- If five clarification questions have been asked and the available evidence still cannot support a useful conclusion, say: "Unfortunately, I do not have enough information to give you a reliable answer."

Insufficient textbook evidence:
- If the provided textbook sources cannot support a useful answer or hypothesis, say: "I can't help you based on the available textbook evidence."
- Do not ask for additional observations when the missing information is in the textbook evidence rather than the user's description.

Textbook sources:
{context}

Previous conversation:
{conversation_text}

Current user message:
{question}

Answer:
"""

    return prompt

def generate_answer(prompt, client, model_name="gpt-6-luna", return_usage=False):
    """Generate an answer and optionally return token usage."""

    response = client.responses.create(
        model=model_name,
        input=prompt
    )

    answer = response.output_text

    # Preserve the original behavior for the chatbot
    if not return_usage:
        return answer

    # Return additional information for evaluation
    usage = response.usage

    return {
        "answer": answer,
        "input_tokens": usage.input_tokens if usage else 0,
        "output_tokens": usage.output_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0}


def ask_tree_assistant(question, model, collection, client, k=5, history=None,
                       previous_results = None, return_results = False):
    """Answer a question using textbook retrieval and an LLM."""

    # taking user question into account
    search_query = question

    # adding earlier questions and infromation
    if history:
        previous_questions = [
            message["content"]
            for message in history
            if message["role"] == "user"
        ]

        if previous_questions:
            original_problem = previous_questions[0]
            search_query = original_problem + " " + question

    # Retrieve relevant textbook passages
    results = search_chroma(
        question=search_query,
        model=model,
        collection=collection,
        k=k
    )

    # combining previous and newly retrieved textbook passages
    if previous_results is not None:
        combined = {
            "ids":[[]],
            "documents":[[]],
            "metadatas":[[]]
        }
        seen_ids = set()
        for source in [results, previous_results]:
            for i, chunk_id in enumerate(source["ids"][0]):
                if chunk_id not in seen_ids:
                    combined["ids"][0].append(chunk_id)
                    combined["documents"][0].append(
                        source["documents"][0][i]
                    )
                    combined["metadatas"][0].append(
                        source["metadatas"][0][i]
                    )
                    seen_ids.add(chunk_id)
        results = combined
    # Build a prompt from the retrieved evidence
    prompt = build_prompt(question, results, history=history)
    # Generate an answer
    answer = generate_answer(prompt, client)

    if return_results:
        return answer, results
    
    return answer

def evaluate_question(question, model, collection, client, k=5, llm_model="gpt-6-luna"):
    """Evaluate one question and record its answer, sources, time, and tokens."""
    import time

    start = time.perf_counter()

    # Retrieves relevant textbook passages
    results = search_chroma(question, model, collection, k=k)

    # Builds the prompt using the retrieved evidence
    prompt = build_prompt(question, results)

    # Generates the answer and collect token usage
    response = generate_answer(
        prompt,
        client,
        model_name=llm_model,
        return_usage=True)

    # Measures elapsed time
    response_time = time.perf_counter() - start

    # Extracts the PDF page numbers of retrieved passages
    retrieved_pages = [
        metadata["pdf_page"]
        for metadata in results["metadatas"][0]]

    return {
        "answer": response["answer"],
        "retrieved_pages": retrieved_pages,
        "response_time": response_time,
        "input_tokens": response["input_tokens"],
        "output_tokens": response["output_tokens"],
        "total_tokens": response["total_tokens"]}