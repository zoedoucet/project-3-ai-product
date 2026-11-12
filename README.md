# Tree Health Investigator

## Overview

This project develops an AI-powered assistant that helps garden and small forest owners understand possible insect-related tree health problems. Users can describe symptoms or observations about their trees and receive information based on a forest entomology book.

The goal of the project is to build a Retrieval-Augmented Generation (RAG) system that retrieves relevant information from the textbook and uses a Large Language Model (LLM) to generate helpful answers. The assistant provides references to the textbook pages used and can ask follow-up questions to help narrow down possible causes.

The application is available online: **[Tree Health Investigator](https://tree-health-investigator.streamlit.app/)**.

---

## Project Structure

The project starts with extracting and cleaning the text from a forest entomology textbook in PDF format. The cleaned text is divided into smaller, overlapping chunks to make it suitable for semantic search.

Next, an embedding model is used to convert these chunks into numerical vectors, which are stored in a ChromaDB vector database. Two embedding models, MiniLM and MPNet, were explored during development. The final application uses MPNet for retrieving relevant textbook passages.

The retrieved passages are then combined with the user's question and passed to an LLM to generate an answer based on the available textbook evidence. The assistant also uses conversation history to follow an investigation and ask relevant clarification questions.

In the final stage, the RAG pipeline is integrated into a Streamlit application, tested, and deployed online using Streamlit Community Cloud.

---

## Requirements

The project uses:

- **Python** for data processing and application development
- **Pandas** for data manipulation
- **Jupyter Notebook** for data exploration and development
- **PyPDF** for extracting text from the textbook
- **Sentence Transformers** for generating text embeddings
- **ChromaDB** for storing and retrieving text embeddings
- **OpenAI API** for generating answers with an LLM
- **Streamlit** for building and deploying the web application
- **Git/GitHub** for version control

---

## How to Run

### 1. Use the online application

The easiest way to use the assistant is through the deployed Streamlit application:

**[Open Tree Health Investigator](https://tree-health-investigator.streamlit.app/)**

Enter a question or describe the symptoms of your tree. The assistant retrieves relevant information from the textbook and generates an answer with page references.

You can provide additional observations or answer follow-up questions to continue the investigation. Use the **New Investigation** button to start a new conversation.

### 2. Clone the repository

Fork and clone the project repository:

[Tree Health Investigator on GitHub](https://github.com/zoedoucet/project-3-ai-product)

### 3. Install the requirements

Install the required Python packages by running:

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

The application requires an OpenAI API key to generate answers.

Create a `.env` file in the project root and add:

```env
OPENAI_API_KEY=your_api_key_here
```

The API key should not be uploaded to GitHub.

### 5. Run the application locally

From the project root, run:

```bash
streamlit run app/Tree_Health_Investigator.py
```

Streamlit will generate a local URL. Open the link in your browser to use the assistant.

To stop the application, return to the terminal and press `Ctrl + C`.

---

## Dataset

For this project, the book *Forest Entomology and Pathology: Volume 1: Entomology* in PDF format was used as the knowledge source.

The textbook contains information about forest insects, their biology, their interactions with trees, and the damage they can cause. This information is used to help the assistant identify possible insect-related problems based on users' observations.

The original PDF contains:

- **810 PDF pages**
- **798 pages with searchable text**
- **3,274 text chunks** after processing
- **150 words per chunk**, with an overlap of 30 words

The text was extracted, cleaned, and divided into overlapping chunks. These chunks were embedded and stored in a ChromaDB vector database together with page information, allowing the assistant to retrieve relevant passages and cite their sources.

---

## Research Questions

The goal of the project is to investigate whether a RAG system can **help users understand possible insect-related tree health problems** by combining semantic search with an LLM.

The main questions are whether the system can retrieve relevant information from a forest entomology textbook, generate understandable answers grounded in this information, and use follow-up questions to narrow down possible causes.

Unlike a traditional classification model, the assistant does not predict a fixed target variable. Instead, it retrieves relevant textbook passages based on the user's observations and generates an answer using this evidence.

---

## Model Performance

The final RAG pipeline uses the **all-mpnet-base-v2** embedding model to retrieve relevant passages from ChromaDB and **GPT-6 Luna** to generate answers.

For each question, the system retrieves the five most relevant text chunks from the textbook. These passages are included in the prompt together with the user's question and conversation history.

The assistant is designed to:

- Provide answers grounded in the retrieved textbook evidence
- Include textbook page references
- Identify possible insect groups or species when supported by the available information
- Ask clarification questions when additional observations could help narrow down the problem
- Communicate uncertainty when the evidence is insufficient

The system was tested with different tree-health questions and follow-up conversations to examine the relevance of the retrieved passages and the quality of the generated answers.

---

## Limitations

The assistant relies on information from a single forest entomology textbook. Its answers are therefore limited by the topics covered in this source and by the relevance of the passages retrieved from the database.

The system focuses on insect-related tree health problems and may not be able to explain symptoms caused by diseases, environmental stress, or other factors.

The quality of the answers also depends on the information provided by the user. Similar symptoms can have different causes, and an exact identification may not be possible without additional observations or an examination of the tree.

Although the assistant is instructed to use only the retrieved textbook evidence, LLM-generated answers can still contain errors. The application is intended as an informational tool rather than a replacement for professional diagnosis.

---

## Acknowledgements

The book used for this project:

Allison, J. D., Paine, T. D., Slippers, B., & Wingfield, M. J. (Eds.). (2023). Forest Entomology and Pathology: Volume 1: Entomology. Springer.

This project was developed as part of the Data Science and Machine Learning Bootcamp at Ironhack.

The application was built with Streamlit and deployed using Streamlit Community Cloud.

The project presentation is available here: [Tree Health Investigator — Presentation](presentation/Tree_Health_Investigator.pdf).