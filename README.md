# HRChatBot

A small HR policy Q&A chatbot. Ask it a question about your company's HR
policy in plain English and it retrieves the relevant policy text and
generates a grounded answer.

Everything runs on **local, free models** - no API keys, no external
services, no per-request cost.

## Architecture

```
hr_policy.txt --> ingest.py --> vector_data/ (texts.pkl + embeddings.npy)
                                        |
                                        v
                question --> chatbot.py --> answer + cited sources
                                        ^
                                        |
                                     app.py (Gradio UI)
```

1. **Ingest** (`ingest.py`) - splits the HR policy document into overlapping
   text chunks and embeds each chunk with a local
   [sentence-transformers](https://www.sbert.net/) model
   (`all-MiniLM-L6-v2`). The chunk texts and their embeddings are saved to
   `vector_data/`.
2. **Retrieve** (`chatbot.py`) - embeds the incoming question with the same
   model and finds the most relevant chunks by cosine similarity, computed
   directly with numpy/scipy (no vector database).
3. **Generate** (`chatbot.py`) - builds a prompt from the retrieved chunks
   and the question, and generates an answer with a local
   [`google/flan-t5-small`](https://huggingface.co/google/flan-t5-small)
   text2text model via Hugging Face `transformers`.
4. **UI** (`app.py`) - a [Gradio](https://www.gradio.app/) interface that
   calls the chatbot functions directly in-process. No separate backend
   server is needed for something this small.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# 1. Build the retrieval index from hr_policy.txt (run once, and again
#    whenever the source document changes)
python ingest.py

# 2. Launch the chatbot UI
python app.py
```

Then open the local URL Gradio prints (typically `http://127.0.0.1:7860`).
Pass `--share` to `app.py` to also get a temporary public link, or
`--help` to see all options.

You can also use the retrieval + generation logic directly from Python:

```python
import chatbot

chatbot.load_index()  # after running ingest.py
result = chatbot.answer_question("When is payroll processed?")
print(result["answer"])
print(result["sources"])
```

## Files

| File | Purpose |
|---|---|
| `hr_policy.txt` | Sample HR policy text. **This is a placeholder** - replace it with your organization's real HR policy document(s) before relying on this for real answers. |
| `ingest.py` | Chunks and embeds the policy text, builds the retrieval index in `vector_data/`. |
| `chatbot.py` | Retrieval (cosine similarity) + generation (flan-t5-small) logic. |
| `app.py` | Gradio UI wired directly to `chatbot.py`. |
| `requirements.txt` | Pinned runtime dependencies. |
| `Chatbot_HR.ipynb` | The original Colab notebook this project started from, kept as an exploration/demo artifact. The `.py` files above are the real, runnable source of truth. |

## Notes / limitations

- The bundled `hr_policy.txt` is a tiny **placeholder** document with five
  made-up policy lines, meant only to demonstrate the pipeline end to end.
  Swap in your real HR policy text (and re-run `ingest.py`) before treating
  any answers as authoritative.
- `flan-t5-small` is a small, fast model chosen so this runs without a GPU
  or API key. Answer quality is modest; swap in a larger local model in
  `chatbot.py` (`GENERATOR_MODEL_NAME`) if you need better answers and have
  the compute for it.
- The first run of `ingest.py` / `app.py` downloads the embedding and
  generation models from Hugging Face (a few hundred MB total) and caches
  them locally; subsequent runs are fast.
