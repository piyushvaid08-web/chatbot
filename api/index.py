"""Vercel serverless entry point for the Nova chatbot.

Each warm serverless instance gets its own in-memory knowledge base and
re-indexes the sample files in data/ on cold start (no embeddings configured
by default, so indexing is fast and free).
"""
import os
import sys

# Make project-root imports (chatbot.*) work inside the serverless bundle.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from chatbot.chat import ChatBot
from chatbot.webapp import create_app

bot = ChatBot()
try:
    from chatbot.config import DATA_DIR, IMAGE_EXTENSIONS

    if os.path.isdir(DATA_DIR):
        docs = []
        from chatbot.loader import load_file

        for name in os.listdir(DATA_DIR):
            path = os.path.join(DATA_DIR, name)
            if (
                os.path.isfile(path)
                and os.path.splitext(name)[1].lower() not in IMAGE_EXTENSIONS
            ):
                try:
                    docs.extend(load_file(path))
                except Exception:
                    pass  # a bad sample file must not kill the deployment
        if docs:
            bot.add_documents(docs)
except Exception:
    pass  # never let startup indexing break the app

app = create_app(bot)
