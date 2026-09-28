"""FastAPI application: serves the chat UI, accepts file uploads, streams replies."""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .chat import ChatBot
from .config import ASSISTANT_NAME, SUPPORTED_EXTENSIONS, UPLOAD_DIR


def create_app(chatbot: ChatBot | None = None) -> FastAPI:
    app = FastAPI(title="File-Aware Chatbot", version="1.0.0")
    bot = chatbot or ChatBot()

    static_dir = Path(__file__).resolve().parent.parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/api/status")
    def status():
        connected = bot.client is not None
        err = getattr(bot, '_client_error', None)
        return {
            "name": ASSISTANT_NAME,
            "connected": connected,
            "files": bot.kb.sources,
            "chunks": len(bot.kb.chunks),
            "error": str(err) if err else None,
        }

    @app.get("/api/files")
    def files():
        return {"files": bot.kb.sources}

    @app.post("/api/upload")
    async def upload(files: List[UploadFile] = File(...)):
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        loaded: List[str] = []
        errors: List[str] = []
        for up in files:
            ext = Path(up.filename or "").suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                errors.append(f"{up.filename}: unsupported type (need one of {', '.join(sorted(SUPPORTED_EXTENSIONS))})")
                continue
            # up.filename can be None for some clients; Path(None) would raise.
            original_name = up.filename or "upload"
            safe_name = f"{uuid.uuid4().hex[:8]}_{Path(original_name).name}"
            dest = os.path.join(UPLOAD_DIR, safe_name)
            with open(dest, "wb") as out:
                out.write(await up.read())
            await up.close()
            try:
                from .loader import load_file

                docs = load_file(dest)
                # Keep the user's original filename as the source label, preserving sheet/page info.
                safe_name_base = os.path.basename(dest)
                renamed_docs = []
                for src, text in docs:
                    new_src = src.replace(safe_name_base, original_name)
                    renamed_docs.append((new_src, text))
                
                bot.kb.add_documents(renamed_docs)
                for source, _text in renamed_docs:
                    if source not in loaded:
                        loaded.append(source)
            except Exception as exc:
                errors.append(f"{up.filename}: {exc}")
        payload = {"loaded": loaded, "errors": errors, "total_chunks": len(bot.kb.chunks)}
        code = 200 if loaded else 400
        return JSONResponse(payload, status_code=code)

    @app.post("/api/chat")
    async def chat(request: dict):
        message = (request.get("message") or "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message is required.")

        def event_stream():
            yield _sse({"role": "assistant", "content": ""})
            try:
                for piece in bot.answer(message, stream=True):
                    yield _sse({"role": "assistant", "content": piece})
            except Exception as exc:
                yield _sse({"error": str(exc)})
            yield _sse({"done": True})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/reset")
    def reset():
        bot.reset()
        return {"ok": True}

    return app


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"