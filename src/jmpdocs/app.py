"""Streamlit chat interface over the JMP documentation.

    streamlit run src/jmpdocs/app.py

Design notes, all driven by how JMP users actually look things up:
  * how-to answers lead with the **menu path**, because that is the first thing
    someone needs before any explanation
  * figures render inline next to the answer, since JMP is a GUI tool and a
    screenshot of the dialog is often the real answer
  * starter prompts are grouped by intent, to solve the blank-page problem
  * an "I'm looking at..." lookup answers from JMP's own context-help map
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jmpdocs.config import get_settings  # noqa: E402
from jmpdocs.ingest.build_index import load_manifest  # noqa: E402
from jmpdocs.ingest.menus import load_menus  # noqa: E402
from jmpdocs.knowledge.aliases import get_alias_bridge  # noqa: E402
from jmpdocs.retrieval.store import get_store, index_exists  # noqa: E402

st.set_page_config(
    page_title="JMP Docs Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

STARTERS = {
    "Get started": [
        "How do I import an Excel file into JMP?",
        "How do I add a new column with a formula?",
        "What is the difference between continuous, nominal and ordinal?",
    ],
    "Analyze": [
        "How do I run a one-way ANOVA to compare group means?",
        "How do I fit a logistic regression?",
        "How do I run a random forest in JMP?",
    ],
    "Visualize": [
        "How do I make a heat map?",
        "How do I build a dashboard?",
        "How do I make a box plot by group?",
    ],
    "DOE": [
        "How do I create a definitive screening design?",
        "I have 5 factors and limited runs, what design should I use?",
        "How do I use the Profiler to optimize multiple responses?",
    ],
    "Quality": [
        "How do I check if my process is capable and get Cpk?",
        "How do I do a Gage R&R study?",
        "How do I make an X-bar and R control chart?",
    ],
    "Scripting": [
        "What does the Choose function do in JSL?",
        "How do I create a window in JSL?",
        "How do I build a JMP add-in?",
    ],
}


# --------------------------------------------------------------------------
# cached resources
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading the JMP documentation index...")
def _store():
    return get_store()


@st.cache_resource(show_spinner=False)
def _graph():
    from jmpdocs.graph.rag_graph import get_graph

    return get_graph()


@st.cache_data(show_spinner=False)
def _ollama_models(base_url: str) -> list[str]:
    import httpx

    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=4)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def _ui_names() -> list[str]:
    import json

    settings = get_settings()
    path = settings.knowledge_dir / "ui_map.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return sorted({u["ui_name"] for u in data})


# --------------------------------------------------------------------------
# rendering helpers
# --------------------------------------------------------------------------


def render_sources(answer, show_scores: bool) -> None:
    if not answer.hits:
        return
    with st.expander(f"Sources ({len(answer.hits)})", expanded=False):
        for i, hit in enumerate(answer.hits, start=1):
            c = hit.chunk
            st.markdown(f"**[{i}] {c.title}**  ·  `{c.page_type}`")
            st.caption(" > ".join(c.breadcrumb) or c.book)
            snippet = c.text[:420] + ("..." if len(c.text) > 420 else "")
            st.markdown(f"> {snippet.replace(chr(10), ' ')}")
            if show_scores:
                st.caption(
                    f"score {hit.score:.3f} · dense {hit.dense_score:.3f} · "
                    f"bm25 {hit.bm25_score:.3f} · rerank {hit.rerank_score:.3f} · "
                    f"boost {hit.boost:.2f}"
                )
            st.markdown(f"[Open in JMP Help]({c.url})")
            if i < len(answer.hits):
                st.divider()


def render_figures(answer) -> None:
    if not answer.figures:
        return
    settings = get_settings()
    shown = []
    for fig in answer.figures[:6]:
        local = settings.images_dir / Path(fig.src).name
        if local.exists():
            shown.append((local, fig))
    if not shown:
        return

    st.markdown("**Figures from the documentation**")
    for row_start in range(0, len(shown), 2):
        cols = st.columns(min(2, len(shown) - row_start))
        for col, (local, fig) in zip(cols, shown[row_start : row_start + 2]):
            with col:
                caption = " - ".join(p for p in (fig.figure_id, fig.caption) if p)
                st.image(str(local), caption=caption or fig.alt, use_container_width=True)


def render_chips(answer) -> None:
    bits: list[str] = []
    if answer.menu_paths:
        bits.append("**Menu:** " + " · ".join(f"`{m}`" for m in answer.menu_paths[:3]))
    if answer.sample_data:
        bits.append("**Sample data:** " + " · ".join(f"`{s}`" for s in answer.sample_data[:3]))
    if answer.alias_terms:
        bits.append("**JMP calls this:** " + " · ".join(f"`{t}`" for t in answer.alias_terms[:3]))
    if bits:
        st.info("  \n".join(bits))


def transcript_markdown() -> str:
    lines = ["# JMP Docs Assistant transcript", ""]
    for turn in st.session_state.get("history", []):
        lines.append(f"## {turn['question']}")
        lines.append("")
        lines.append(turn["answer"])
        if turn.get("sources"):
            lines.append("")
            lines.append("**Sources**")
            for i, s in enumerate(turn["sources"], start=1):
                lines.append(f"{i}. [{s['title']}]({s['url']}) - {s['breadcrumb']}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------


def sidebar(settings) -> dict:
    with st.sidebar:
        st.markdown("### JMP Docs Assistant")

        manifest = load_manifest(settings)
        if manifest:
            st.caption(
                f"JMP {manifest.get('jmp_version')} · "
                f"{manifest.get('n_chunks', 0):,} chunks · "
                f"{manifest.get('n_pages', 0):,} pages"
            )
            st.caption(f"Index built {manifest.get('built_at', 'unknown')}")

        st.divider()

        models = _ollama_models(settings.llm.base_url)
        if models:
            default = (
                models.index(settings.llm.answer_model)
                if settings.llm.answer_model in models
                else 0
            )
            answer_model = st.selectbox("Answer model", models, index=default)
            fast_default = (
                models.index(settings.llm.fast_model)
                if settings.llm.fast_model in models
                else default
            )
            fast_model = st.selectbox(
                "Fast model",
                models,
                index=fast_default,
                help="Used for query rewriting on follow-up turns. A small model here "
                "cuts noticeable latency.",
            )
        else:
            answer_model = fast_model = settings.llm.answer_model
            st.warning(f"Ollama not reachable at {settings.llm.base_url}")

        st.divider()

        try:
            books = _store().books
        except Exception:
            books = []
        book_filter = st.multiselect("Limit to books", books, default=[])

        k = st.slider("Excerpts to retrieve", 3, 12, settings.retrieval.k_final)
        show_scores = st.toggle("Show retrieval scores", value=False)

        st.divider()
        st.markdown("**I'm looking at...**")
        names = _ui_names()
        picked = st.selectbox(
            "On-screen element",
            ["(none)"] + names[:4000],
            index=0,
            label_visibility="collapsed",
            help="Names taken from JMP's own context-sensitive help map.",
        )
        lookup = None
        if picked != "(none)" and st.button("Explain this element", use_container_width=True):
            lookup = picked

        st.divider()
        if st.session_state.get("history"):
            st.download_button(
                "Export transcript",
                transcript_markdown(),
                file_name="jmp-docs-transcript.md",
                mime="text/markdown",
                use_container_width=True,
            )
        if st.button("Clear chat", use_container_width=True):
            st.session_state.history = []
            st.session_state.thread_id = f"t{time.time():.0f}"
            st.rerun()

        bridge = get_alias_bridge()
        st.caption(f"{len(bridge)} vocabulary aliases · {len(load_menus())} menu items")

    return {
        "answer_model": answer_model,
        "fast_model": fast_model,
        "books": book_filter,
        "k": k,
        "show_scores": show_scores,
        "lookup": lookup,
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
    settings = get_settings()

    if not index_exists(settings):
        st.title("JMP Docs Assistant")
        st.error("The documentation index has not been built yet.")
        st.code(
            "python scripts/build_corpus.py\npython scripts/build_index.py",
            language="bash",
        )
        st.stop()

    st.session_state.setdefault("history", [])
    st.session_state.setdefault("thread_id", f"t{time.time():.0f}")
    st.session_state.setdefault("pending", None)

    opts = sidebar(settings)

    # apply model overrides for this run
    settings.llm.answer_model = opts["answer_model"]
    settings.llm.fast_model = opts["fast_model"]
    settings.retrieval.k_final = opts["k"]

    st.title("JMP Docs Assistant")
    st.caption(
        "Ask about JMP 19.1 in your own words. Answers come only from the official "
        "documentation, with sources."
    )

    if opts["lookup"]:
        st.session_state.pending = f"What is the {opts['lookup']} in JMP?"

    # starter prompts, shown only on an empty chat
    if not st.session_state.history and not st.session_state.pending:
        st.markdown("#### Try one of these")
        tabs = st.tabs(list(STARTERS))
        for tab, (group, prompts) in zip(tabs, STARTERS.items()):
            with tab:
                for p in prompts:
                    if st.button(p, key=f"start-{group}-{p}", use_container_width=True):
                        st.session_state.pending = p
                        st.rerun()

    # replay history
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])

    typed = st.chat_input("Ask about JMP...")
    question = typed or st.session_state.pending
    st.session_state.pending = None

    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        from jmpdocs.graph.rag_graph import astream_answer

        placeholder = st.empty()
        started = time.time()
        buffer: list[str] = []
        answer = None

        try:
            stream = astream_answer(
                question,
                thread_id=st.session_state.thread_id,
                books=opts["books"] or None,
                settings=settings,
            )
            for kind, payload in stream:
                if kind == "token":
                    buffer.append(payload)
                    placeholder.markdown("".join(buffer) + "▌")
                else:
                    answer = payload
        except Exception as exc:  # Ollama down, model missing, etc.
            placeholder.error(f"{type(exc).__name__}: {exc}")
            return

        text = "".join(buffer).strip() or (answer.answer if answer else "")
        placeholder.markdown(text or "_No answer produced._")
        st.caption(
            f"{time.time() - started:.1f}s · intent: "
            f"`{answer.intent if answer else 'unknown'}` · model: `{opts['answer_model']}`"
        )

        if answer:
            render_chips(answer)
            render_figures(answer)
            render_sources(answer, opts["show_scores"])

            st.session_state.history.append(
                {
                    "question": question,
                    "answer": text,
                    "sources": [
                        {
                            "title": h.chunk.title,
                            "url": h.chunk.url,
                            "breadcrumb": " > ".join(h.chunk.breadcrumb),
                        }
                        for h in answer.hits
                    ],
                }
            )


main()
