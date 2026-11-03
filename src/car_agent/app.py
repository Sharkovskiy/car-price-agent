"""Gradio chat UI. Tool calls are shown as collapsible steps above the answer."""
from __future__ import annotations

import argparse
import json
import uuid

import gradio as gr

from car_agent.graph import CarAgent, build_agent

EXAMPLES = [
    "Скільки коштує Toyota RAV4 2016 року з пробігом 80 тисяч миль у Каліфорнії?",
    "Мені пропонують Honda Civic 2017, 70 тис. миль, Нью-Йорк, за $16,500. Це нормальна ціна?",
    "Find me a family SUV under $15,000, 2014 or newer, in Texas. Show the best deals.",
    "Знайди седан до $10,000 з невеликим пробігом у Флориді, тільки від приватних власників",
]


def _short(text: str, limit: int = 1_500) -> str:
    """Pretty-print a JSON tool result for the collapsible step, truncated for readability."""
    try:
        text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except ValueError:
        pass
    return text if len(text) <= limit else text[:limit] + "\n..."


def create_demo(agent: CarAgent, data_snapshot: str | None) -> gr.ChatInterface:
    # One conversation thread per browser session; clearing the chat starts a new thread.
    threads: dict[str, str] = {}

    def respond(message: str, history: list, request: gr.Request):
        session = request.session_hash if request else "local"
        if not history or session not in threads:
            threads[session] = uuid.uuid4().hex
        thread_id = threads[session]

        steps: list[gr.ChatMessage] = []
        try:
            for event in agent.stream(message, thread_id=thread_id):
                if event.kind == "tool_call":
                    steps.append(gr.ChatMessage(
                        content=f"```json\n{json.dumps(event.args, ensure_ascii=False, indent=2)}\n```",
                        metadata={"title": f"Tool: {event.name}", "status": "pending"},
                    ))
                elif event.kind == "tool_result":
                    pending = next((s for s in steps if s.metadata.get("status") == "pending"
                                    and s.metadata.get("title") == f"Tool: {event.name}"), None)
                    if pending:
                        pending.content += f"\n\nResult:\n```json\n{_short(event.content)}\n```"
                        pending.metadata["status"] = "done"
                elif event.kind == "answer":
                    steps.append(gr.ChatMessage(content=event.content))
                yield steps
        except Exception as error:  # show API problems (e.g. missing key, rate limit) in the chat
            steps.append(gr.ChatMessage(content=f"Error: {type(error).__name__}: {error}"))
            yield steps

    return gr.ChatInterface(
        fn=respond,
        title="Used-car price assistant",
        description=(
            "Price estimates and search over U.S. Craigslist listings "
            f"({data_snapshot or '2021'}). Ask in any language."
        ),
        examples=EXAMPLES,
        fill_height=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the car price agent chat UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a temporary public *.gradio.live link")
    args = parser.parse_args()

    agent, store = build_agent()
    create_demo(agent, store.data_snapshot).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
