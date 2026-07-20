"""PyInstaller entry point used by the Tauri backend sidecar."""
import argparse
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()
uvicorn.run("app.main:app", host="127.0.0.1", port=args.port)
