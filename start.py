#!/usr/bin/env python3
"""Cross-platform setup and start script for the Multi-Agent Research System.

Works on macOS, Linux, and Windows. Requires Python 3.12+.

Usage:
    python start.py          # Setup .env, install deps, start server
    python start.py --setup  # Only setup .env and install deps
    python start.py --run    # Only start the server (skip setup)
"""

from __future__ import annotations

import getpass
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"

REQUIRED_KEYS = [
    ("ANTHROPIC_API_KEY", "Anthropic API key (for Claude LLMs)"),
    ("TAVILY_API_KEY", "Tavily API key (for web search)"),
]


def print_banner() -> None:
    print()
    print("=" * 58)
    print("   Multi-Agent Research System - Setup & Start")
    print(f"   Python {sys.version.split()[0]}  |  {platform.system()} {platform.machine()}")
    print("=" * 58)
    print()


def check_python_version() -> None:
    if sys.version_info < (3, 12):
        print(f"ERROR: Python 3.12+ is required. You have {sys.version}")
        sys.exit(1)


def setup_env() -> None:
    """Copy .env.example to .env and prompt for required keys."""
    if not ENV_EXAMPLE.exists():
        print("ERROR: .env.example not found at project root.")
        sys.exit(1)

    if ENV_FILE.exists():
        print(f"Found existing .env file at {ENV_FILE}")
        answer = input("Overwrite with fresh copy from .env.example? [y/N]: ").strip().lower()
        if answer != "y":
            print("Keeping existing .env — checking for missing keys...\n")
            patch_missing_keys()
            return

    shutil.copy2(ENV_EXAMPLE, ENV_FILE)
    print(f"Copied .env.example -> .env\n")
    patch_missing_keys()


def read_env_file() -> dict[str, str]:
    """Parse .env file into a dict (ignores comments and blank lines)."""
    values: dict[str, str] = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def write_key_to_env(key: str, value: str) -> None:
    """Replace a key's placeholder value in the .env file."""
    content = ENV_FILE.read_text()
    new_lines = []
    found = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(new_lines) + "\n")


def is_placeholder(value: str) -> bool:
    """Check if a value looks like a placeholder rather than a real key."""
    placeholders = {"", "your_anthropic_api_key", "your_tavily_api_key", "your_openai_api_key"}
    return value.lower() in placeholders or value.startswith("your_")


def patch_missing_keys() -> None:
    """Prompt the user for any required keys that are missing or still placeholders."""
    current = read_env_file()
    for key, description in REQUIRED_KEYS:
        existing = current.get(key, "")
        if not is_placeholder(existing):
            print(f"  {key}: already set")
            continue
        print(f"  {key} is required ({description})")
        value = getpass.getpass(f"  Enter {key}: ").strip()
        if not value:
            print(f"  ERROR: {key} cannot be empty. Exiting.")
            sys.exit(1)
        write_key_to_env(key, value)
        print(f"  {key}: saved to .env")
    print()


def install_deps() -> None:
    """Install project dependencies."""
    print("Installing dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".[dev]"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    print()


def start_server() -> None:
    """Start the FastAPI server with uvicorn."""
    print("Starting server on http://localhost:8000")
    print("Press Ctrl+C to stop.\n")
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "src.server:app", "--reload", "--port", "8000"],
        cwd=PROJECT_ROOT,
    )


def main() -> None:
    print_banner()
    check_python_version()

    args = set(sys.argv[1:])
    run_only = "--run" in args
    setup_only = "--setup" in args

    if not run_only:
        setup_env()
        install_deps()
        print("Setup complete!\n")

    if setup_only:
        print("Run 'python start.py --run' to start the server.")
        return

    start_server()


if __name__ == "__main__":
    main()
