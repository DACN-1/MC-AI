"""Native inference server for split MineRL rollouts.

Runs on the host (e.g. macOS + Metal/MPS), loads a VLA checkpoint via the
gym-free `agent_loader`, and answers per-frame inference requests over HTTP.
The MineRL env stays in the Docker container and calls this with
`run_rollout.py --remote-agent host.docker.internal:<port>`; the model forward
then runs on the host GPU instead of the container's (emulated) CPU.

Endpoints:
  GET  /config   -> {"past_action_k", "chunk_size", "use_language"}
  POST /predict  <- {"pov": <base64 raw uint8>, "shape": [H,W,3],
                     "prompt": str, "past": [float, ...]}
                 -> {"logits": [[...43...], ... chunk_size ...]}

Single-threaded on purpose: the rollout loop issues one request per env step,
and MPS does not like concurrent use. Sequential serving keeps it simple.
"""

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch
from PIL import Image

from agent_loader import load_agent


def _resolve_device(requested: str) -> str:
    if requested == "mps" and not torch.backends.mps.is_available():
        print("⚠️  MPS requested but unavailable; falling back to CPU.")
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return requested


def build_handler(agent, agent_cfg, device):
    class _Handler(BaseHTTPRequestHandler):
        # Per-process request counter for a lightweight heartbeat.
        n_requests = 0
        # HTTP/1.1 enables connection keep-alive, which is essential when the
        # client is on the other end of a high-latency tunnel (TCP+SSH
        # handshake per request adds ~1.5 s, dominating wall-clock). Pair with
        # `Connection: keep-alive` from _RemoteAgent.
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):  # silence default per-request logging
            pass

        def _send_json(self, obj, status=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/config":
                self._send_json(agent_cfg)
            elif self.path == "/health":
                self._send_json({"ok": True, "device": device})
            else:
                self._send_json({"error": "unknown path"}, status=404)

        def do_POST(self):
            if self.path != "/predict":
                self._send_json({"error": "unknown path"}, status=404)
                return
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length).decode("utf-8"))

            # Prefer the JPEG-compressed payload from new clients (saves ~15x
            # on the wire — see _RemoteAgent in run_rollout.py); fall back to
            # the legacy raw uint8 base64 path for older clients.
            if "pov_jpeg" in req:
                import io as _io
                img = Image.open(_io.BytesIO(base64.b64decode(req["pov_jpeg"]))).convert("RGB")
            else:
                arr = np.frombuffer(base64.b64decode(req["pov"]), dtype=np.uint8)
                arr = arr.reshape(req["shape"])
                img = Image.fromarray(arr)
            prompt = req.get("prompt", "")
            past = req.get("past") or []
            past_tensor = (
                torch.tensor(past, dtype=torch.float32).reshape(1, -1) if past else None
            )

            with torch.no_grad():
                logits = agent([img], [prompt], past_tensor)  # (1, chunk, 43)
            logits = logits[0].detach().to("cpu").tolist()

            _Handler.n_requests += 1
            if _Handler.n_requests % 100 == 0:
                print(f"served {_Handler.n_requests} predictions", flush=True)

            self._send_json({"logits": logits})

    return _Handler


def main():
    p = argparse.ArgumentParser(description="Native VLA inference server.")
    p.add_argument("--model-path", required=True, help="Path to checkpoint (.pt)")
    p.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="mps / cuda / cpu (default: mps if available)",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    device = _resolve_device(args.device)
    print(f"Loading {args.model_path} on {device} …", flush=True)
    agent, agent_cfg = load_agent(args.model_path, device)
    print(f"Loaded. config={agent_cfg}", flush=True)

    server = HTTPServer((args.host, args.port), build_handler(agent, agent_cfg, device))
    print(f"Serving on {args.host}:{args.port} (device={device}). Ctrl-C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
