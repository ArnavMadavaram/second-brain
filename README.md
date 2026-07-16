# second-brain
Personal RAG with an emotion-aware embedding model


# Second Brain

A personal retrieval system that answers questions about your own life using your past writing. It ingests personal text, mainly an exported ChatGPT history and journal entries, and lets you ask natural-language questions that are answered from your own words.

The distinctive part of this project is the retrieval step. Instead of a general-purpose embedding model, it uses a custom embedding model fine-tuned to understand emotional similarity, and evaluates whether that improves retrieval of emotionally relevant memories compared to a standard model.

> Status: in development. This README is updated as the project is built.

---

## What It Does

- Takes personal text (ChatGPT export, journal entries) and makes it searchable by meaning and emotion
- Answers questions like "when have I felt most uncertain about my future" using your own past writing
- Tags each entry with emotional information (valence, arousal, and a discrete emotion)
- Shows which memories an answer was based on, for transparency
- Includes a lightweight daily check-in to keep adding new entries over time

---

## Why It Exists

Most people have years of personal writing scattered across apps with no good way to search it by meaning or emotion. Standard tools match on keywords and cannot reliably surface, for example, the entry where you were anxious about a decision unless you used the word anxious.

This project asks a specific technical question: does an embedding model fine-tuned for emotional similarity retrieve more relevant personal memories than a standard off-the-shelf model? The answer is measured, not assumed.

The broad area is affective computing. The tagging step is emotion classification. The valence-arousal representation is dimensional emotion analysis.

---

## How It Works

The system has two flows.

### Ingestion (runs when data goes in)

1. **Parse** the ChatGPT export JSON into clean conversation records
2. **Filter** personal, emotionally meaningful content from task content (homework, debugging)
3. **Chunk** the kept text into pieces with overlap so context is preserved
4. **Tag** each chunk with emotion, mapped to valence and arousal
5. **Embed** each chunk into a vector using the custom fine-tuned model
6. **Store** the vector, original text, and tags in the vector database

### Query (runs when you ask something)

1. **Embed** the question with the same custom model
2. **Search** the vector store for the closest stored memories
3. **Retrieve** the top matches
4. **Respond** by passing those memories and the question to the language model
5. **Show** the answer alongside the memories it was based on

The custom embedding model is used in both flows and is the core contribution.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Backend | FastAPI |
| Vector storage | ChromaDB (local) |
| Structured storage | SQLite |
| Base embedding model | all-MiniLM-L6-v2, fine-tuned |
| Fine-tuning | sentence-transformers |
| Emotion tagging | pretrained GoEmotions classifier |
| Response generation | Claude |
| Frontend | simple web app |

Everything runs locally. Personal data stays on the user's machine. Only the small amount of retrieved text needed to answer a question is sent to the language model.

---

## Project Structure

_To be filled in as the project takes shape._

```
second-brain/
  (structure will be documented here as it is built)
```

---

## Setup

_To be filled in as setup steps are finalized. Will include environment setup, dependencies, how to provide an API key, and how to run the ingestion and the app._

```
# placeholder
```

---

## Usage

_To be filled in once the interface is working. Will cover how to load data, ask questions, and use the daily check-in._

---

## Evaluation

The core research claim is evaluated by comparing the custom fine-tuned embedding model against a standard baseline model, measuring retrieval quality on emotional queries. Details and results will be documented here as the evaluation is built and run.

---

## Design Decisions

Key decisions and the reasoning behind them are documented as the project develops. A few settled so far:

- **ChromaDB over IndexedDB** for vector storage, to keep the project in Python and reusable across future clients rather than locked to the browser
- **A pretrained emotion classifier, not an LLM, for tagging**, to avoid circular LLM-in, LLM-out processing
- **FastAPI over Flask or Django**, as a good fit for an ML backend with slow operations
- **Local-first storage**, since the data is deeply personal

---

## Limitations and Future Work

Known limitations and features intentionally left out of the current scope because they need long-term accumulated data:

- Temporal retrieval that weighs recent and recurring memories differently
- A causal graph mapping how feelings, circumstances, and decisions connect
- A decision prediction study
- A proactive insight layer
- A mobile app (the backend is kept client-agnostic to allow this later)

---

## Acknowledgements

Developed as a CS494 Directed Study at Purdue University Fort Wayne, with guidance from Professor Thomas Bolinger (Computer Science) and input on the emotional framework from Professor Raymond Voss (Psychology).
