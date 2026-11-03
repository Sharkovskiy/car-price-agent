"""Agent loop tests with a scripted chat model (no API calls)."""
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from car_agent.config import Settings
from car_agent.graph import CarAgent, context_messages


class ScriptedChatModel(BaseChatModel):
    """Returns pre-defined messages one by one and records what it was sent."""

    responses: list = Field(default_factory=list)
    received: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])

    def bind_tools(self, tools, **kwargs):
        return self


class FakeTools:
    def __init__(self):
        self.calls = []

    def run(self, name, args, max_chars=6000):
        self.calls.append((name, args))
        return json.dumps({"price": 18285, "price_low": 13405, "price_high": 24677})


def tool_call(name, args, call_id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def make_agent(responses):
    llm = ScriptedChatModel(responses=responses)
    tools = FakeTools()
    agent = CarAgent(tools, Settings(artifacts_dir=None), data_snapshot="2021-04-04 .. 2021-05-05", llm=llm)
    return agent, llm, tools


def test_tool_call_then_answer():
    agent, llm, tools = make_agent([
        tool_call("estimate_price", {"manufacturer": "toyota", "model": "rav4", "year": 2016}),
        AIMessage(content="<think>hidden</think>About $18,285."),
    ])
    turn = agent.ask("How much is a 2016 RAV4?", thread_id="t1")

    assert turn.answer == "About $18,285."  # reasoning block removed
    assert [c.name for c in turn.tool_calls] == ["estimate_price"]
    assert json.loads(turn.tool_calls[0].output)["price"] == 18285
    assert tools.calls == [("estimate_price", {"manufacturer": "toyota", "model": "rav4", "year": 2016})]
    # The second LLM call sees the tool result.
    assert isinstance(llm.received[1][-1], ToolMessage)


def test_follow_up_keeps_history_but_drops_old_tool_traffic():
    agent, llm, _ = make_agent([
        tool_call("estimate_price", {"year": 2016}),
        AIMessage(content="About $18,285."),
        AIMessage(content="With more miles it is cheaper."),
    ])
    agent.ask("How much is a 2016 RAV4?", thread_id="t2")
    agent.ask("And with 120k miles?", thread_id="t2")

    sent = llm.received[-1]
    contents = [m.content for m in sent[1:]]  # skip the system prompt
    assert contents == ["How much is a 2016 RAV4?", "About $18,285.", "And with 120k miles?"]
    assert not any(isinstance(m, ToolMessage) for m in sent)


def test_threads_are_independent():
    agent, llm, _ = make_agent([AIMessage(content="A"), AIMessage(content="B")])
    agent.ask("first", thread_id="x")
    agent.ask("second", thread_id="y")
    assert [m.content for m in llm.received[-1][1:]] == ["second"]


def test_context_messages_limits_history():
    messages = []
    for i in range(10):
        messages += [HumanMessage(f"q{i}"), AIMessage(content=f"a{i}")]
    messages.append(HumanMessage("current"))
    context = context_messages(messages, max_history_turns=2)
    assert [m.content for m in context] == ["q8", "a8", "q9", "a9", "current"]
