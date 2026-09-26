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

SYSTEM_PROMPT = f"""You are {ASSISTANT_NAME}, an AI assistant that answers questions about the user's uploaded documents.

Answering from documents (most important):
- Relevant excerpts are given below each user message, labelled [Source: ...].
  Base your answer on them.
- Quote figures exactly. Double-check numbers against the excerpts before
  answering. Never invent data.
- If the question names a specific sheet, section or period (e.g. "Q1", "page
  2"), use ONLY the rows from that scope — do not pull in other sheets.
- For calculations (totals, averages, counts, highest/lowest), work through
  them step by step, show the numbers you added, and state the final result
  clearly. Re-check your arithmetic before answering.
- If you use a document, mention where it came from, e.g. "according to
  sales.xlsx [Sheet: Q1]" or "on page 3 of guide.pdf".
- If the excerpts don't contain the answer, say you don't have that
  information instead of guessing.

Tone:
- Be warm and conversational, like a smart friend: use contractions and vary
  your sentence structure.
- Match the user's language (Spanish, Hindi, French, etc.).
- Be concise: answer the question directly first, then add helpful detail.
- If the user's question is casual (greetings, small talk, opinions), just chat
  naturally — no need to dig into documents."""


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
        self.kb = KnowledgeBase()
        self._client_error = None
        if api_key:
            try:
                self.client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=20.0,
                    max_retries=1,
                    default_headers={"OpenAI-Beta": "embeddings=2"},
                )
            except Exception as exc:
                # Client creation failed (e.g. SSL, DNS) — stay offline
                self._client_error = exc
                self.client = None
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
        if hasattr(self, '_client_error') and self._client_error:
            return (
                f"I can't reach my AI brain — the connection failed: {self._client_error}. "
                f"Check your network and try again."
            )
        return (
            f"I don't have my brain connected yet — no OPENAI_API_KEY found in .env. "
            f"Drop your key into a .env file (see .env.example) and restart me. "
            f"In the meantime, I've loaded {len(self.kb.chunks)} chunks from your files "
            f"and can still answer from them once the key is set."
        )

    def _build_messages(self, user_message: str):
        context = self._build_context(user_message)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history[-MAX_HISTORY * 2 :])
        messages.append({"role": "user", "content": context + "\n\n" + user_message})
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
            reply = (
                f"Sorry, I hit a snag talking to my brain: {exc}. "
                f"Check OPENAI_API_KEY in your .env file and try again."
            )
            self._client_error = exc
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
                f"Check OPENAI_API_KEY in your .env file and try again."
            )
            return

        parts: List[str] = []
        for event in response:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            piece = (delta or {}).get("content") if isinstance(delta, dict) else (delta.content if delta else None)
            if piece:
                parts.append(piece)
                yield piece
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": "".join(parts)})

    def _build_context(self, user_message: str) -> str:
        """Retrieve relevant chunks and format them as prompt context."""
        results = self.kb.search(user_message, top_k=TOP_K)
        if not results:
            return "(No documents loaded yet, or nothing relevant found.)"
        blocks = []
        for idx, (source, text, _score) in enumerate(results, start=1):
            snippet = text if len(text) <= 1800 else text[:1800] + " …"
            blocks.append(f"[Excerpt {idx} | Source: {source}]\n{snippet}")
        return "Relevant excerpts from your documents:\n\n" + "\n\n".join(blocks)


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