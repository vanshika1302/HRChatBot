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
"""

import argparse

import chatbot


def get_answer(question, k):
    if not question or not question.strip():
        return "Please enter a question."
    try:
        result = chatbot.answer_question(question, k=int(k))
    except FileNotFoundError as e:
        return str(e)

    answer = result["answer"]
    sources = result["sources"]
    source_lines = "\n".join(
        f"- [Context {s['idx'] + 1}] (score={s['score']:.2f}): {s['text']}"
        for s in sources
    )
    return f"{answer}\n\nSources:\n{source_lines}"


def build_interface():
    import gradio as gr

    return gr.Interface(
        fn=get_answer,
        inputs=[
            gr.Textbox(label="Ask a question about the HR policy"),
            gr.Slider(minimum=1, maximum=5, value=2, step=1, label="Number of source chunks (k)"),
        ],
        outputs=gr.Textbox(label="Answer"),
        title="HR Policy Chatbot",
        description=(
            "Ask questions about the sample HR policy. Retrieval + generation "
            "run entirely on local models (sentence-transformers + flan-t5-small) "
            "- no API keys, no external services."
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the HR policy chatbot Gradio UI.")
    parser.add_argument("--share", action="store_true", help="Create a public gradio.live share link")
    parser.add_argument("--server-port", type=int, default=None, help="Port to serve on")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    iface = build_interface()
    iface.launch(share=args.share, server_port=args.server_port)
