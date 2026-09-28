"""The chat engine.

Builds a conversation around the knowledge base: every user message is used to
retrieve the most relevant chunks of the uploaded documents, which are handed
to the LLM as context. The system prompt pushes for warm, human, natural
replies — and honest "I don't know" when nothing relevant is found.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from .config import (
    ASSISTANT_NAME,
    MAX_HISTORY,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    TOP_K,
)
from .loader import RawDocument
from .memory import KnowledgeBase

SYSTEM_PROMPT = f"""You are {ASSISTANT_NAME}, a personal AI assistant possessing vast general knowledge about the world. You are exceptionally smart, helpful, and versatile.

General Knowledge:
- Answer any question the user has using your expansive knowledge base.
- Be highly informative, yet concise.

Document Analysis (if excerpts are provided):
- Base your answers on the provided excerpts when the question relates to them.
- Quote figures exactly. Double-check numbers against the excerpts before answering.
- Mention where data came from, e.g. "according to sales.xlsx [Sheet: Q1]".

Tone:
- Be warm and conversational, like a smart friend.
- Match the user's language (Spanish, Hindi, French, etc.)."""


class ChatBot:
    def __init__(
        self,
        api_key: str = OPENAI_API_KEY,
        base_url: str = OPENAI_BASE_URL,
        model: str = OPENAI_MODEL,
    ):
        self.api_key = api_key
        self.model = model
        self.client = None
        self._client_error: Optional[Exception] = None
        if api_key:
            try:
                self.client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=20.0,
                    max_retries=1,
                )
            except Exception as exc:
                # Client creation failed (e.g. SSL, DNS) — stay offline
                self._client_error = exc
                self.client = None
        else:
            self._client_error = RuntimeError(
                "OPENAI_API_KEY is not set (add it to .env and restart)."
            )
        # Pass the client so document chunks actually get embedded; otherwise
        # the knowledge base silently runs in keyword-only mode forever.
        self.kb = KnowledgeBase(client=self.client)
        self.history: List[Dict[str, str]] = []

    # -- documents --------------------------------------------------------------
    def load_files(self, paths) -> List[str]:
        """Load files/directories into the knowledge base; returns source labels."""
        documents: List[RawDocument] = []
        for raw in paths:
            p = os.path.abspath(raw)
            if os.path.isdir(p):
                from .loader import load_paths

                documents.extend(load_paths([p]))
            else:
                from .loader import load_file

                documents.extend(load_file(p))
        self.kb.add_documents(documents)
        return list(self.kb.sources)

    def add_documents(self, documents: List[RawDocument]) -> int:
        return self.kb.add_documents(documents)

    def reset(self) -> None:
        """Forget documents and conversation."""
        self.kb.clear()
        self.history.clear()

    # -- chat --------------------------------------------------------------------
    def answer(self, user_message: str, stream: bool = False):
        """Get a reply as text. If ``stream`` is True, returns a generator that
        yields text fragments instead."""
        if stream:
            return self._answer_stream(user_message)
        return self._answer_plain(user_message)

    def _no_key_message(self) -> str:
        if isinstance(self._client_error, RuntimeError):
            return (
                "I don't have my brain connected yet — no OPENAI_API_KEY found in .env. "
                "Drop your key into a .env file (see .env.example) and restart me. "
                f"In the meantime, I've loaded {len(self.kb.chunks)} chunks from your files "
                "and can still search them once the key is set."
            )
        if self._client_error:
            return (
                "I can't reach my AI brain — the connection failed: "
                f"{self._client_error}. Check your network and try again."
            )
        return "My AI brain is unavailable right now. Please try again."

    def _build_messages(self, user_message: str):
        context = self._build_context(user_message)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history[-MAX_HISTORY * 2 :])
        messages.append({"role": "user", "content": context + user_message})
        return messages

    def _answer_plain(self, user_message: str) -> str:
        if not self.client:
            return self._no_key_message()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(user_message),
                temperature=0.3,
                max_tokens=700,
                stream=False,
            )
            reply = _content_text(response.choices[0].message.content)
        except Exception as exc:  # network error, bad key, over-quota...
            self._client_error = exc
            reply = (
                f"Sorry, I hit a snag talking to my brain: {exc}. "
                "Check OPENAI_API_KEY in your .env file and try again."
            )
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def _answer_stream(self, user_message: str):
        if not self.client:
            yield self._no_key_message()
            return
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(user_message),
                temperature=0.3,
                max_tokens=700,
                stream=True,
            )
        except Exception as exc:
            self._client_error = exc
            yield (
                f"Sorry, I hit a snag talking to my brain: {exc}. "
                "Check OPENAI_API_KEY in your .env file and try again."
            )
            return

        parts: List[str] = []
        try:
            for event in response:
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                # delta is a Pydantic model (ChoiceDelta), NOT a plain dict —
                # always use attribute access, never .get()
                piece = getattr(delta, "content", None)
                if piece:
                    parts.append(piece)
                    yield piece
        except Exception as exc:  # dropped connection mid-stream
            piece = f"\n\n[Stream interrupted: {exc}]"
            parts.append(piece)
            yield piece
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": "".join(parts)})

    def _build_context(self, user_message: str) -> str:
        """Retrieve relevant chunks and format them as prompt context."""
        results = self.kb.search(user_message, top_k=TOP_K)
        if not results:
            return ""
        blocks = []
        for idx, (source, text, _score) in enumerate(results, start=1):
            snippet = text if len(text) <= 1800 else text[:1800] + " …"
            blocks.append(f"[Excerpt {idx} | Source: {source}]\n{snippet}")
        return "Relevant excerpts from your documents:\n\n" + "\n\n".join(blocks) + "\n\n"


def _content_text(content) -> str:
    """Normalize message content to plain text (openai SDKs may return a string
    or a list of content parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            text = getattr(part, "text", None) or (part.get("text") if isinstance(part, dict) else None)
            if text:
                parts.append(text)
        return "".join(parts)
    return str(content or "")
