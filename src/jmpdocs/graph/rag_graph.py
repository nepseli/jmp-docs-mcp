"""The LangGraph RAG pipeline.

    route ─┬─(chitchat)──────────────────────────────────► generate
           └─(doc question)─► classify ─► expand ─► retrieve ─► grade ─┐
                                            ▲                          │
                                            └──── retry (broadened) ◄──┘
                                                                       ▼
                                                        generate ─► figures

Latency is the binding constraint here, not cost: generation runs on CPU, so
every LLM call costs seconds. The graph is therefore built to *skip* LLM calls
it does not need:

  * routing and intent use free deterministic rules
  * query expansion only runs when it can actually help -- a follow-up turn
    with pronouns to resolve, or a very short question
  * the retry only fires when retrieval reports nothing relevant was found

That keeps a typical single-turn question to one LLM call (the answer itself).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Annotated, Any, Iterator, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from jmpdocs.config import Settings, get_settings
from jmpdocs.graph.prompts import (
    GRADE_PROMPT,
    QUERY_EXPANSION_PROMPT,
    system_prompt,
)
from jmpdocs.ingest.parse import Figure
from jmpdocs.knowledge.aliases import get_alias_bridge
from jmpdocs.knowledge.intent import classify_intent
from jmpdocs.retrieval.hybrid import Hit, search
from jmpdocs.retrieval.store import get_store

# Greetings and meta-questions that need no retrieval at all.
_CHITCHAT_RE = re.compile(
    r"^\s*(hi|hey|hello|yo|thanks|thank you|ta|cheers|bye|goodbye|"
    r"who are you|what can you do|what do you do|help)\b[\s!.?]*$",
    re.I,
)


class RagState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    route: str
    intent: str
    queries: list[str]
    alias_terms: list[str]
    book_filter: list[str]
    hits: list[Hit]
    figures: list[Figure]
    attempts: int
    context: str
    top_relevance: float
    relevance_floor: float


@dataclass(slots=True)
class RagAnswer:
    answer: str
    intent: str
    hits: list[Hit] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    alias_terms: list[str] = field(default_factory=list)
    menu_paths: list[str] = field(default_factory=list)
    sample_data: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def make_llm(model: str, settings: Settings | None = None, **kwargs: Any):
    from langchain_ollama import ChatOllama

    st = settings or get_settings()
    return ChatOllama(
        model=model,
        base_url=st.llm.base_url,
        temperature=st.llm.temperature,
        num_ctx=st.llm.num_ctx,
        num_predict=st.llm.num_predict,
        # qwen3 reasons by default, which doubles latency for no gain here --
        # the retrieved excerpts already carry the reasoning.
        reasoning=st.llm.reasoning,
        **kwargs,
    )


def format_context(hits: Sequence[Hit], max_chars: int | None = None) -> str:
    """Number the excerpts so the model can cite them as [1], [2]."""
    if max_chars is None:
        max_chars = get_settings().llm.max_context_chars
    parts: list[str] = []
    used = 0
    for i, h in enumerate(hits, start=1):
        c = h.chunk
        header = f"[{i}] {c.title} - {c.context_line}"
        if c.menu_paths:
            header += f"\n    Menu: {'; '.join(c.menu_paths[:3])}"
        body = c.text
        block = f"{header}\n{body}"
        if used + len(block) > max_chars:
            block = block[: max(0, max_chars - used)]
            if block:
                parts.append(block)
            break
        parts.append(block)
        used += len(block)
    return "\n\n---\n\n".join(parts)


def _history(messages: Sequence[BaseMessage], limit: int = 6) -> str:
    lines = []
    for m in list(messages)[-limit:]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        text = m.content if isinstance(m.content, str) else str(m.content)
        lines.append(f"{role}: {text[:400]}")
    return "\n".join(lines)


_PROSE_RE = re.compile(
    r"(the user|i need|i'?ll|let me|first,|okay|so they|this means|"
    r"looking at|conversation|assistant)", re.I
)


def _looks_like_query(line: str) -> bool:
    """Keep search queries, drop the reasoning prose models like to emit."""
    line = line.strip()
    if not (3 < len(line) <= 90):
        return False
    if _PROSE_RE.search(line):
        return False
    return len(line.split()) <= 14 and not line.endswith((".", "!", "?:"))


def _needs_expansion(question: str, messages: Sequence[BaseMessage]) -> bool:
    """Only pay for an LLM rewrite when it can plausibly help."""
    has_history = sum(1 for m in messages if isinstance(m, HumanMessage)) > 1
    if has_history and re.search(
        r"\b(it|that|this|those|these|them|there|same|instead|also|"
        r"what about|how about)\b",
        question,
        re.I,
    ):
        return True
    return len(question.split()) <= 3


# --------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------


def build_graph(settings: Settings | None = None, checkpointer: Any | None = None):
    st = settings or get_settings()
    bridge = get_alias_bridge()

    def node_route(state: RagState) -> RagState:
        question = state.get("question", "").strip()
        route = "chitchat" if _CHITCHAT_RE.match(question) else "docs"
        return {"route": route, "attempts": 0}

    def node_classify(state: RagState) -> RagState:
        question = state["question"]
        return {
            "intent": state.get("intent") or classify_intent(question),
            "alias_terms": bridge.jmp_terms_for(question),
        }

    def node_expand(state: RagState) -> RagState:
        """Resolve follow-up questions into self-contained search queries.

        Deterministic by default. An LLM rewrite was measured on the available
        local models and rejected: qwen3 ignores both `reasoning=False` and the
        `/no_think` token, so it spent 31-36 s emitting first-person reasoning
        prose instead of queries -- slower than the answer itself, and actively
        harmful once those prose lines were used as search queries.

        Carrying the previous question forward does the one thing that actually
        mattered (resolving "it" / "that platform") for free. Set
        llm.use_llm_expansion if a model that honours no-think is available.
        """
        question = state["question"]
        messages = state.get("messages", [])
        queries = [question]

        if _needs_expansion(question, messages):
            prior = [
                m.content
                for m in messages[:-1]
                if isinstance(m, HumanMessage) and isinstance(m.content, str)
            ]
            if prior:
                # "and how do I add statistics to it?" alone retrieves nothing;
                # paired with the previous turn it retrieves Tabulate.
                queries.append(f"{prior[-1]} {question}")

        if st.llm.use_llm_expansion and _needs_expansion(question, messages):
            llm = make_llm(st.llm.fast_model, st)
            prompt = (
                f"{QUERY_EXPANSION_PROMPT}\n\n"
                f"Conversation:\n{_history(messages)}\n\n"
                f"Latest question: {question}"
            )
            try:
                raw = llm.invoke([HumanMessage(content=prompt)]).content
                text = raw if isinstance(raw, str) else str(raw)
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
                queries.extend(
                    line for line in
                    (l.strip(" -*\t0123456789.") for l in text.splitlines())
                    if _looks_like_query(line)
                )
            except Exception:
                pass  # expansion is an optimisation, never a hard dependency

        return {"queries": list(dict.fromkeys(q for q in queries if q))[:4]}

    def node_retrieve(state: RagState) -> RagState:
        attempts = state.get("attempts", 0)
        queries = list(state.get("queries") or [state["question"]])
        # a retry widens the net: drop the intent boost and take more candidates
        result = search(
            state["question"],
            queries=queries,
            intent=None if attempts else state.get("intent"),
            books=state.get("book_filter") or None,
            k=st.retrieval.k_final + (3 if attempts else 0),
            settings=st,
        )
        figures: list[Figure] = []
        seen: set[str] = set()
        for h in result.hits:
            for f in h.chunk.figures:
                if f.src not in seen:
                    seen.add(f.src)
                    figures.append(f)
        return {
            "hits": result.hits,
            "figures": figures,
            "context": format_context(result.hits),
            "attempts": attempts + 1,
            "top_relevance": result.top_relevance,
            "relevance_floor": result.relevance_floor,
        }

    def node_generate(state: RagState) -> RagState:
        llm = make_llm(st.llm.answer_model, st)
        if state.get("route") == "chitchat":
            msgs: list[BaseMessage] = [
                SystemMessage(
                    content=(
                        "You are a JMP 19.1 documentation assistant. Introduce "
                        "yourself in one or two sentences and invite a question "
                        "about JMP. Do not invent JMP details."
                    )
                ),
                HumanMessage(content=state["question"]),
            ]
        else:
            msgs = [
                SystemMessage(content=system_prompt(state.get("intent", "how_to"))),
                *state.get("messages", [])[:-1],
                HumanMessage(
                    content=(
                        f"Documentation excerpts:\n\n{state.get('context', '')}\n\n"
                        f"Question: {state['question']}"
                    )
                ),
            ]
        reply = llm.invoke(msgs)
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        return {"messages": [AIMessage(content=text)]}

    # ---- wiring ---------------------------------------------------------

    def after_route(state: RagState) -> str:
        return "generate" if state.get("route") == "chitchat" else "classify"

    def after_retrieve(state: RagState) -> str:
        hits = state.get("hits") or []
        attempts = state.get("attempts", 0)
        weak = not hits or state.get("top_relevance", 1.0) < state.get("relevance_floor", 0.45)
        if weak and attempts <= st.llm.max_retrieval_retries:
            return "retrieve"
        return "generate"

    graph = StateGraph(RagState)
    graph.add_node("route", node_route)
    graph.add_node("classify", node_classify)
    graph.add_node("expand", node_expand)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("generate", node_generate)

    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", after_route, ["classify", "generate"])
    graph.add_edge("classify", "expand")
    graph.add_edge("expand", "retrieve")
    graph.add_conditional_edges("retrieve", after_retrieve, ["retrieve", "generate"])
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())


# --------------------------------------------------------------------------
# convenience API
# --------------------------------------------------------------------------


_GRAPH = None


def get_graph(settings: Settings | None = None):
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph(settings)
    return _GRAPH


def _collect(state: dict) -> RagAnswer:
    hits = state.get("hits") or []
    messages = state.get("messages") or []
    answer = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            answer = m.content if isinstance(m.content, str) else str(m.content)
            break

    menu_paths: list[str] = []
    samples: list[str] = []
    for h in hits:
        for m in h.chunk.menu_paths:
            if m not in menu_paths:
                menu_paths.append(m)
        for s in h.chunk.sample_data:
            if s not in samples:
                samples.append(s)

    return RagAnswer(
        answer=answer,
        intent=state.get("intent", "how_to"),
        hits=hits,
        figures=state.get("figures") or [],
        queries=state.get("queries") or [],
        alias_terms=state.get("alias_terms") or [],
        menu_paths=menu_paths[:5],
        sample_data=samples[:5],
    )


def ask(
    question: str,
    *,
    thread_id: str = "default",
    books: Sequence[str] | None = None,
    settings: Settings | None = None,
) -> RagAnswer:
    """Run the full graph and return the answer with its provenance."""
    graph = get_graph(settings)
    state = graph.invoke(
        {
            "question": question,
            "messages": [HumanMessage(content=question)],
            "book_filter": list(books or []),
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    return _collect(state)


def astream_answer(
    question: str,
    *,
    thread_id: str = "default",
    books: Sequence[str] | None = None,
    settings: Settings | None = None,
) -> Iterator[tuple[str, Any]]:
    """Stream ('token', str) then a final ('done', RagAnswer).

    Retrieval runs first so the UI can show sources while the answer streams.
    """
    st = settings or get_settings()
    graph = get_graph(st)
    config = {"configurable": {"thread_id": thread_id}}

    final_state: dict = {}
    for mode, payload in graph.stream(
        {
            "question": question,
            "messages": [HumanMessage(content=question)],
            "book_filter": list(books or []),
        },
        config=config,
        stream_mode=["messages", "values"],
    ):
        if mode == "messages":
            chunk, meta = payload
            # Only the incremental chunks. LangGraph emits BOTH the streamed
            # AIMessageChunks and, once the node returns, the complete AIMessage
            # it wrote to state -- accepting both renders every answer twice.
            if (
                isinstance(chunk, AIMessageChunk)
                and meta.get("langgraph_node") == "generate"
                and chunk.content
            ):
                yield "token", chunk.content
        elif mode == "values":
            final_state = payload

    yield "done", _collect(final_state)
