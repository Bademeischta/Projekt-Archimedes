#!/usr/bin/env python3
"""
Start script: install deps, then run training and dashboard in parallel.
Detects Colab vs local; in Colab starts Ngrok for public dashboard URL.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)

    # Install dependencies
    print("Installing dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        check=False,
        cwd=root,
    )

    use_public = in_colab() or os.environ.get("ARCHIMEDES_PUBLIC_DASHBOARD", "").lower() in ("1", "true", "yes")
    port = 8501

    # Start training in background
    train_cmd = [
        sys.executable, "train_end_to_end.py",
        "--total-games", "20",
        "--training-iterations", "50",
        "--num-workers", "1",
        "--checkpoint-dir", ".",
    ]
    print("Starting training in background:", " ".join(train_cmd))
    train_proc = subprocess.Popen(
        train_cmd,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    # Start Streamlit in background
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", "dashboard.py",
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    print("Starting Streamlit on port", port, "...")
    dash_proc = subprocess.Popen(
        streamlit_cmd,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    time.sleep(3)

    if use_public:
        try:
            from pyngrok import ngrok
            url = ngrok.connect(port, "http")
            print("\n" + "=" * 60)
            print("Dashboard (public):", url.public_url)
            print("=" * 60 + "\n")
        except Exception as e:
            print("Ngrok failed:", e)
            print("Dashboard (local): http://localhost:" + str(port))
    else:
        print("Dashboard (local): http://localhost:" + str(port))
        print("Set ARCHIMEDES_PUBLIC_DASHBOARD=1 for public URL via Ngrok.")

    print("Press Ctrl+C to stop training and dashboard.")
    try:
        train_proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        train_proc.terminate()
        dash_proc.terminate()
        try:
            train_proc.wait(timeout=5)
            dash_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            train_proc.kill()
            dash_proc.kill()

if __name__ == "__main__":
    main()
