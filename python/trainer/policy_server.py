# python/trainer/policy_server.py

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from flask import Flask, request, jsonify
import logging

from model import SnakeDQN  # adjust import if needed

app = Flask(__name__)

logging.getLogger("werkzeug").setLevel(logging.ERROR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net: Optional[SnakeDQN] = None
HEIGHT: Optional[int] = None
WIDTH: Optional[int] = None


def load_checkpoint(path: str):
    global policy_net, HEIGHT, WIDTH

    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device)

    cfg_dict = checkpoint.get("config", {})
    height = cfg_dict.get("height", None)
    width = cfg_dict.get("width", None)

    # Fallback: try to infer from state dict if not stored in config
    if height is None or width is None:
        raise ValueError
        

    HEIGHT = int(height)
    WIDTH = int(width)

    policy_net = SnakeDQN(height=height, width=width, num_actions=4).to(device)
    policy_net.load_state_dict(checkpoint["model_state_dict"])
    policy_net.eval()

    print(f"Loaded model with height={height}, width={width}")


@app.route("/act", methods=["POST"])
def act():
    """
    Expects JSON like:
    {
      "grid": [0,1,0,...],   // length = width * height
      "width": 10,
      "height": 10
    }
    Returns:
    {
      "action": 0   // int in {0,1,2,3}
    }
    """
    global policy_net, HEIGHT, WIDTH

    if policy_net is None:
        return jsonify({"error": "model not loaded"}), 500

    try:
        data: Dict[str, Any] = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid JSON"}), 400

    grid = data.get("grid")
    width = int(data.get("width", 0))
    height = int(data.get("height", 0))

    if grid is None or not isinstance(grid, list):
        return jsonify({"error": "grid missing or invalid"}), 400

    if width <= 0 or height <= 0:
        return jsonify({"error": "width/height must be > 0"}), 400

    arr = np.asarray(grid, dtype=np.int8)
    if arr.size != width * height:
        return jsonify({"error": "grid size mismatch"}), 400

    # Ensure we match the training dimensions (you can relax this if you support multiple sizes)
    if HEIGHT is not None and WIDTH is not None:
        if width != WIDTH or height != HEIGHT:
            return jsonify({"error": "wrong board size for this model"}), 400

    arr = arr.reshape((height, width))

    # Convert to tensor: (1, 1, H, W)
    state_t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = policy_net(state_t)
        action = int(q_values.argmax(dim=1).item())

    return jsonify({"action": action})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,   # <-- was True
        help="Path to .pt checkpoint file saved by training loop",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6000,
    )

    args = parser.parse_args()

    checkpoint = args.checkpoint
    if not checkpoint:
        checkpoint = os.getenv("CHECKPOINT_PATH")

    if not checkpoint:
        raise SystemExit(
            "No checkpoint provided. "
            "Pass --checkpoint or set CHECKPOINT_PATH env var."
        )

    load_checkpoint(checkpoint)

    print(f"Starting policy server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
