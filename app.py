"""
app.py

Gradio UI for the HR policy chatbot. Calls chatbot.answer_question()
directly in-process - no separate FastAPI/backend hop needed for
something this small.

Usage:
    python ingest.py         # build the retrieval index (once, or whenever
                              # hr_policy.txt changes)
    python app.py            # launch a local Gradio server
    python app.py --share    # also create a public gradio.live link
    python app.py --help

Environment variables:
    SCREENSHOT_MODE=1        # DEV/DOCS ONLY - see the block below.
                              # Stubs chatbot.answer_question() with a
                              # canned response so the real UI can be
                              # screenshotted without needing to download
                              # model weights. Never set this for normal use.
"""

import argparse
import os

# Disable Gradio's phone-home analytics ping - keeps things quiet (and
# functional) in network-restricted environments, and is good practice for
# a local-only demo anyway.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import chatbot


# ---------------------------------------------------------------------------
# SCREENSHOT_MODE - used ONLY for generating documentation / portfolio
# screenshots in environments where downloading the real embedding and
# generation model weights from Hugging Face isn't possible (e.g. a
# network-restricted sandbox). When active, it monkeypatches
# chatbot.answer_question() with a canned, realistic-looking response so the
# *real* Gradio UI (theme, layout, chat rendering, examples, etc.) can still
# be exercised end-to-end and captured in a screenshot.
#
# This is NOT part of normal application behavior:
#   - it is entirely inert unless the SCREENSHOT_MODE env var is set to "1"
#   - it is never set by default, in requirements.txt, or in CI
#   - it must never be relied on for anything other than taking screenshots
# ---------------------------------------------------------------------------
if os.environ.get("SCREENSHOT_MODE") == "1":

    def _canned_answer_question(question, k=2, **kwargs):
        """Fake stand-in for chatbot.answer_question(), screenshots only."""
        return {
            "answer": (
                "Employees are entitled to 20 days of paid annual leave per "
                "calendar year [Context 1]. Leave requests should be "
                "submitted through the HR portal and approved by your "
                "manager in advance."
            ),
            "sources": [
                {
                    "text": "Employees are entitled to 20 days of paid annual leave.",
                    "score": 0.91,
                    "idx": 0,
                },
                {
                    "text": "Work from home is allowed 2 days per week with manager approval.",
                    "score": 0.47,
                    "idx": 1,
                },
            ],
        }

    chatbot.answer_question = _canned_answer_question
    print(
        "[SCREENSHOT_MODE] chatbot.answer_question() is stubbed with a "
        "canned response for documentation screenshots only - this is not "
        "the real model output."
    )
# ---------------------------------------------------------------------------


HOW_IT_WORKS = """
This assistant answers questions using a small **retrieval-augmented
generation (RAG)** pipeline that runs entirely on local, free models -
no API keys, no external services, no per-request cost.

1. **Embed** - `ingest.py` splits the HR policy document into overlapping
   chunks and embeds each one with a local `sentence-transformers` model
   (`all-MiniLM-L6-v2`), saving the result to `vector_data/`.
2. **Retrieve** - your question is embedded with that same model, and the
   most similar chunks are found by cosine similarity - computed directly
   with numpy/scipy, no vector database involved.
3. **Generate** - the retrieved chunks are inserted into a prompt and a
   local `google/flan-t5-small` model generates an answer grounded in
   them, citing the context it drew from (e.g. `[Context 1]`).

```
hr_policy.txt --> ingest.py --> vector_data/ (chunks + embeddings)
                                        |
              question --> retrieve (cosine similarity)
                                        |
                                 generate (flan-t5-small)
                                        |
                                 answer + cited sources
```
"""

EXAMPLE_QUESTIONS = [
    "How many days of paid annual leave do I get?",
    "When is payroll processed?",
    "Can I work from home, and how often?",
    "Do I need a doctor's note for sick leave?",
]


def respond(message, history, k):
    """gr.ChatInterface callback - wired directly to chatbot.answer_question()
    (or its SCREENSHOT_MODE stub above, when that's active)."""
    message = (message or "").strip()
    if not message:
        return "Please enter a question."

    try:
        result = chatbot.answer_question(message, k=int(k))
    except FileNotFoundError as e:
        return str(e)

    answer = result["answer"]
    sources = result.get("sources") or []
    if sources:
        source_lines = "\n".join(
            f"- **[Context {s['idx'] + 1}]** &middot; relevance {s['score']:.2f} &mdash; {s['text']}"
            for s in sources
        )
        return f"{answer}\n\n---\n**Sources**\n{source_lines}"
    return answer


def build_interface():
    import gradio as gr

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
    )

    with gr.Blocks(
        theme=theme, title="HR Policy Assistant", fill_height=True, analytics_enabled=False
    ) as demo:
        gr.Markdown(
            """
            # 🗂️ HR Policy Assistant
            Ask a question about your company's HR policy in plain English.
            Answers are retrieved from the policy document and generated by a
            small local language model - grounded in the source text, with
            citations, and running entirely offline.
            """
        )

        with gr.Accordion("How it works (RAG architecture)", open=False):
            gr.Markdown(HOW_IT_WORKS)

        gr.ChatInterface(
            fn=respond,
            type="messages",
            chatbot=gr.Chatbot(
                label="HR Policy Assistant",
                type="messages",
                height=420,
                show_copy_button=True,
                avatar_images=(None, None),
            ),
            textbox=gr.Textbox(
                placeholder="e.g. How many vacation days do I have?",
                scale=7,
            ),
            additional_inputs=[
                gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                    label="Source chunks to retrieve (k)",
                    info="How many policy passages to retrieve before generating an answer.",
                ),
            ],
            additional_inputs_accordion="Advanced options",
            examples=[[q] for q in EXAMPLE_QUESTIONS],
        )

        gr.Markdown(
            "<sub>Portfolio/demo project. The bundled HR policy is a small "
            "placeholder document - swap in a real policy before relying on "
            "any answers.</sub>"
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the HR policy chatbot Gradio UI.")
    parser.add_argument("--share", action="store_true", help="Create a public gradio.live share link")
    parser.add_argument("--server-port", type=int, default=None, help="Port to serve on")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = build_interface()
    demo.launch(share=args.share, server_port=args.server_port)
