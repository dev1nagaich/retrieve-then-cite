from __future__ import annotations

from pathlib import Path

import gradio as gr

from rag_app.config import DEFAULT_INDEX_DIR
from rag_app.pipeline import RAGPipeline


def launch_ui(*, index_dir: Path = DEFAULT_INDEX_DIR, host: str = "127.0.0.1", port: int = 7860) -> None:
    pipeline = RAGPipeline(index_dir=index_dir)

    def answer_query(question: str, top_k: int):
        if not question.strip():
            return "Please enter a question.", "", []

        result = pipeline.answer(question.strip(), top_k=int(top_k))
        citations = "\n".join(f"- {citation}" for citation in result["citations"])
        rows = [
            [
                item["book"],
                item["pdf_page"],
                item["chunk_id"],
                round(float(item["score"]), 4) if item["score"] is not None else None,
                item["preview"],
            ]
            for item in result["retrieved"]
        ]
        return result["answer"], citations, rows

    with gr.Blocks(title="Sports Medicine RAG") as demo:
        gr.Markdown("# Sports Medicine Retrieve-Then-Cite RAG")
        gr.Markdown("Paste a query and get a grounded answer with book/page citations.")
        with gr.Row():
            query = gr.Textbox(
                label="Question",
                placeholder="Example: What are common acute concussion symptoms?",
                lines=4,
            )
        top_k = gr.Slider(1, 10, value=5, step=1, label="Top K retrieved chunks")
        submit = gr.Button("Answer", variant="primary")
        answer = gr.Textbox(label="Answer", lines=10)
        citations = gr.Markdown(label="Citations")
        retrieved = gr.Dataframe(
            headers=["Book", "PDF page", "Chunk", "Score", "Preview"],
            label="Retrieved passages",
            wrap=True,
        )
        submit.click(answer_query, inputs=[query, top_k], outputs=[answer, citations, retrieved])

    demo.launch(server_name=host, server_port=port)
