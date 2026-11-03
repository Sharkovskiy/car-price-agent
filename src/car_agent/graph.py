"""LangGraph agent: an LLM node decides, a tool node executes, looped until the LLM answers in text.

    START -> agent --(tool calls)--> tools -> agent -> ... -> agent --(text)--> END

Conversation memory is kept by a checkpointer per thread_id. To save tokens, the LLM sees the
full tool exchange only for the current user message; from earlier turns it sees just the user
messages and the final answers.
"""
from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from car_agent.config import Settings
from car_agent.prompts import system_prompt
from car_agent.tool_schemas import tool_schemas


class AgentState(MessagesState):
    tool_rounds: int


@dataclass
class ToolCallRecord:
    name: str
    args: dict
    output: str = ""


@dataclass
class AgentTurn:
    question: str
    answer: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class AgentEvent:
    kind: str                 # "tool_call" | "tool_result" | "answer"
    name: str = ""
    args: dict | None = None
    content: str = ""


def strip_reasoning(text: Any) -> str:
    """Remove <think>...</think> blocks in case a reasoning model still returns them."""
    if isinstance(text, list):  # content blocks
        text = "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in text)
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def context_messages(messages: list[BaseMessage], max_history_turns: int) -> list[BaseMessage]:
    """Messages sent to the LLM: earlier turns without tool traffic + the current turn in full."""
    last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
    history = [
        m for m in messages[:last_human]
        if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)
    ]
    return history[-2 * max_history_turns:] + messages[last_human:]


def make_llm(settings: Settings) -> BaseChatModel:
    from langchain_groq import ChatGroq

    options: dict[str, Any] = {"reasoning_format": "hidden"}
    if settings.groq_model.startswith("openai/gpt-oss"):
        options["reasoning_effort"] = settings.reasoning_effort
    return ChatGroq(
        model=settings.groq_model,
        temperature=settings.temperature,
        max_tokens=settings.max_completion_tokens,
        max_retries=6,  # exponential backoff, including free-tier rate limits (429)
        **options,
    )


class CarAgent:
    def __init__(self, tools: Any, settings: Settings, data_snapshot: str | None = None,
                 llm: BaseChatModel | None = None):
        """tools: object with run(name, args, max_chars) -> str (see car_agent.tools.CarTools)."""
        self.tools = tools
        self.settings = settings
        self.prompt = system_prompt(data_snapshot)
        self.llm = llm or make_llm(settings)
        schemas = tool_schemas(settings.max_results)
        self.llm_with_tools = self.llm.bind_tools(schemas)
        self.llm_final = self.llm.bind_tools(schemas, tool_choice="none")
        self.graph = self._build_graph()

    # ---- graph -------------------------------------------------------------------
    def _call_model(self, state: AgentState) -> dict:
        rounds = state.get("tool_rounds", 0)
        messages = [SystemMessage(self.prompt), *context_messages(state["messages"], self.settings.max_history_turns)]
        # Tool budget exhausted: the model must answer with what it has collected.
        model = self.llm_with_tools if rounds < self.settings.max_tool_rounds else self.llm_final
        response = model.invoke(messages)
        truncated = response.response_metadata.get("finish_reason") == "length"
        if truncated and not response.tool_calls:
            # Usually hidden reasoning used up the budget: retry once with twice as many tokens.
            response = model.bind(max_tokens=2 * self.settings.max_completion_tokens).invoke(messages)
        return {"messages": [response]}

    def _call_tools(self, state: AgentState) -> dict:
        last = state["messages"][-1]
        outputs = [
            ToolMessage(
                content=self.tools.run(call["name"], call["args"], self.settings.max_tool_result_chars),
                tool_call_id=call["id"],
                name=call["name"],
            )
            for call in last.tool_calls
        ]
        return {"messages": outputs, "tool_rounds": state.get("tool_rounds", 0) + 1}

    @staticmethod
    def _route(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("agent", self._call_model)
        builder.add_node("tools", self._call_tools)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", self._route, {"tools": "tools", END: END})
        builder.add_edge("tools", "agent")
        return builder.compile(checkpointer=MemorySaver())

    # ---- public API --------------------------------------------------------------
    def stream(self, question: str, thread_id: str = "default") -> Iterator[AgentEvent]:
        """Run one user turn and yield tool calls, tool results and the final answer as they happen."""
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 2 * self.settings.max_tool_rounds + 6,
        }
        inputs = {"messages": [HumanMessage(question)], "tool_rounds": 0}
        for update in self.graph.stream(inputs, config, stream_mode="updates"):
            for node, values in update.items():
                for message in values.get("messages", []):
                    if node == "agent" and message.tool_calls:
                        for call in message.tool_calls:
                            yield AgentEvent("tool_call", name=call["name"], args=call["args"])
                    elif node == "agent":
                        yield AgentEvent("answer", content=strip_reasoning(message.content))
                    elif node == "tools":
                        yield AgentEvent("tool_result", name=message.name, content=message.content)

    def ask(self, question: str, thread_id: str = "default") -> AgentTurn:
        """Run one user turn and return the answer together with the tool calls (used by evaluation)."""
        started = time.time()
        turn = AgentTurn(question=question, answer="")
        for event in self.stream(question, thread_id):
            if event.kind == "tool_call":
                turn.tool_calls.append(ToolCallRecord(event.name, event.args or {}))
            elif event.kind == "tool_result":
                pending = next((c for c in turn.tool_calls if c.name == event.name and not c.output), None)
                if pending:
                    pending.output = event.content
            elif event.kind == "answer":
                turn.answer = event.content
        turn.seconds = time.time() - started
        return turn


def build_agent(settings: Settings | None = None) -> tuple[CarAgent, Any]:
    """Load artifacts and create the agent. Returns (agent, store)."""
    from car_agent.config import load_settings
    from car_agent.store import load_store
    from car_agent.tools import CarTools

    settings = settings or load_settings()
    store = load_store(settings.artifacts_dir)
    tools = CarTools(store, max_results=settings.max_results)
    return CarAgent(tools, settings, data_snapshot=store.data_snapshot), store
