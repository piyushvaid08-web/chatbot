#!/usr/bin/env python3
"""Entry point for the file-aware chatbot.

Web UI (default):
    python main.py --web

CLI chat:
    python main.py                    # auto-loads files from data/
    python main.py data/report.xlsx   # loads specific files/dirs first
"""
import argparse
import os
import sys

from chatbot.config import DATA_DIR, UPLOAD_DIR


def run_cli(paths) -> None:
    from chatbot.chat import ChatBot

    bot = ChatBot()
    docs_to_load = [p for p in paths if os.path.exists(p)]
    for d in (DATA_DIR, UPLOAD_DIR):
        if os.path.isdir(d) and d not in docs_to_load:
            docs_to_load.append(d)
    _load_startup_files(bot, docs_to_load)

    print()
    print("========================================")
    print("  Nova - file-aware assistant")
    print("  Type your question. /reset = forget files+history.")
    print("  Ctrl+C to quit.")
    print("========================================")
    print()

    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye! ")
            break
        if not question:
            continue
        if question.lower() in {"/reset", "/clear"}:
            bot.reset()
            print("Nova> Memory cleared.")
            continue
        if question.lower() in {"/quit", "/exit", "quit", "exit"}:
            print("Nova> Bye! ")
            break
        if question.lower() in {"/files", "/sources"}:
            print("Nova> Loaded sources: " + (", ".join(bot.kb.sources) if bot.kb.sources else "none yet"))
            continue

        print("Nova> ", end="", flush=True)
        try:
            for piece in bot.answer(question, stream=True):
                print(piece, end="", flush=True)
        except Exception as exc:
            print(f"[error: {exc}]", end="")
        print("\n")


def run_web(port: int, paths) -> None:
    from chatbot.chat import ChatBot
    from chatbot.webapp import create_app

    bot = ChatBot()
    loadable = [p for p in paths if os.path.exists(p)]
    for d in (DATA_DIR, UPLOAD_DIR):
        if os.path.isdir(d) and d not in loadable:
            loadable.append(d)
    _load_startup_files(bot, loadable, quiet=True)

    import uvicorn

    app = create_app(bot)
    print(f"\nNova is running - open http://localhost:{port} in your browser")
    uvicorn.run(app, host="0.0.0.0", port=port)


def _load_startup_files(bot, paths, quiet: bool = False) -> None:
    """Load startup documents, skipping images (API-billed and useless to
    pre-index) and reporting failures without killing the app."""
    from chatbot.config import IMAGE_EXTENSIONS

    filtered = []
    for p in paths:
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in IMAGE_EXTENSIONS:
            continue
        filtered.append(p)
    if not filtered:
        return
    try:
        sources = bot.load_files(filtered)
        if sources and not quiet:
            print(f"Loaded {len(sources)} source(s):")
            for s in sources:
                print(f"   - {s}")
    except Exception as exc:
        print(f"Could not load files: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="File-aware chatbot (web UI or CLI).")
    parser.add_argument("paths", nargs="*", help="files or directories to load (e.g. data/ report.xlsx)")
    parser.add_argument("--web", action="store_true", help="start the web UI instead of the CLI")
    parser.add_argument("--port", type=int, default=8000, help="port for the web UI (default: 8000)")
    args = parser.parse_args()

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    if args.web:
        run_web(args.port, args.paths)
    else:
        run_cli(args.paths)


if __name__ == "__main__":
    sys.exit(main())