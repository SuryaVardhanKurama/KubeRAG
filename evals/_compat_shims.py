"""Compatibility shims for ragas 0.4.x on langchain-community >= 0.3.

ragas 0.4.3's `ragas/llms/base.py` hard-imports two symbols that were removed
from langchain-community in v0.3.0 (Vertex AI integrations moved to the
standalone `langchain-google-vertexai` package):

    from langchain_community.chat_models.vertexai import ChatVertexAI
    from langchain_community.llms import VertexAI

The eval pipeline only ever constructs the OpenAI provider
(`llm_factory(provider="openai", ...)`), so these Vertex AI classes are
imported but never instantiated. We pre-register in-memory shims that
re-export the real classes from `langchain_google_vertexai` — nothing in
site-packages is modified and nothing outside this project changes.

This module must be imported BEFORE any `ragas` import (see evals/__init__.py).
"""
import sys
import types


def _install() -> None:
    try:
        from langchain_google_vertexai import ChatVertexAI, VertexAI
    except ImportError:
        # langchain_google_vertexai not installed — let ragas fail naturally
        return

    # `from langchain_community.chat_models.vertexai import ChatVertexAI`
    if "langchain_community.chat_models.vertexai" not in sys.modules:
        chat_mod = types.ModuleType("langchain_community.chat_models.vertexai")
        chat_mod.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = chat_mod

    # `from langchain_community.llms import VertexAI`
    try:
        import langchain_community.llms as lc_llms

        if not hasattr(lc_llms, "VertexAI"):
            lc_llms.VertexAI = VertexAI
    except ImportError:
        pass


_install()
