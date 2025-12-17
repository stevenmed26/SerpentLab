# python/trainer/train_service.py

import os
import json
import threading
import time
import queue
from dataclasses import asdict

from flask import Flask, request, jsonify, Response

from python.trainer.DQN.train_loop import TrainConfig, train

app = Flask(__name__)

_train_thread = None
_stop_flag = threading.Event()
_status_lock = threading.Lock()

_status = {
    "running": False,
    "episode": 0,
    "last_reward": None,
    "avg_last_50": None,
    "foods": None,
    "message": "",
}

_metrics_q: "queue.Queue[dict]" = queue.Queue(maxsize=1000)


def push_metric(m: dict):
    try:
        _metrics_q.put_nowait(m)
    except queue.Full:
        try:
            _metrics_q.get_nowait()
        except queue.Empty:
            pass
        try:
            _metrics_q.put_nowait(m)
        except queue.Full:
            pass

def set_status(**kwargs):
    with _status_lock:
        _status.update(kwargs)


def get_status():
    with _status_lock:
        return dict(_status)
    

def train_runner(cfg: TrainConfig):
    set_status(running=True, message="training started")
    _stop_flag.clear()

    try:
        train(cfg, stop_event=_stop_flag, on_episode=push_metric, set_status=set_status)
        set_status(message="training finished")
    except Exception as e:
        set_status(message=f"training crashed: {e}")
    finally:
        set_status(running=False)


@app.post("/start")
def start():
    global _train_thread
    if _train_thread and _train_thread.is_alive():
        return jsonify({"ok": False, "error": "training already running"}), 409
    
    data = request.get_json(force=True, silent=True) or {}

    cfg = TrainConfig(
        address=data.get("address", os.getenv("TRAINER_ENV_ADDR", "env-gateway: 50051")),
        width=int(data.get("width", 10)),
        height=int(data.get("height", 10)),
        with_walls=bool(data.get("with_walls", True)),

        num_episodes=int(data.get("num_episodes", 10_000)),
        max_steps_per_episode=int(data.get("max_steps_per_episode", 500)),

        batch_size=int(data.get("batch_size", 64)),
        gamma=float(data.get("gamma", 0.99)),

        eps_start=float(data.get("eps_start", 1.0)),
        eps_end=float(data.get("eps_end", 0.10)),
        eps_decay_episodes=int(data.get("eps_decay_episodes", 2500)),

        buffer_capacity=int(data.get("buffer_capacity", 100_000)),
        learning_rate=float(data.get("learning_rate", 1e-3)),
        target_update_interval=int(data.get("target_update_interval", 50)),

        checkpoint_dir=data.get("checkpoint_dir", "../models/checkpoints"),
        checkpoint_interval=int(data.get("checkpoint_interval", 1000)),
        # Optional: resume
        resume_from=data.get("resume_from", ""),
    )

    set_status(running=False, episode=0, last_reward=None, avg_last_50=None, foods=None, message="starting...")
    _train_thread = threading.Thread(target=train_runner, args=(cfg,), daemon=True)
    _train_thread.start()

    return jsonify({"ok": True, "config": asdict(cfg)})


@app.post("/stop")
def stop():
    if not (_train_thread and _train_thread.is_alive()):
        return jsonify({"ok": True, "message": "not running"})
    _stop_flag.set()
    return jsonify({"ok": True, "message": "stop requested"})


@app.get("/status")
def status():
    return jsonify({"ok": True, "status": get_status()})



@app.get("/metrics")
def metrics():
    # Server-Sent Events stream
    def gen():
        yield "event: hello\ndata: {}\n\n"
        while True:
            try:
                m = _metrics_q.get(timeout=1.0)
                yield f"data: {json.dumps(m)}\n\n"
            except queue.Empty:
                # keep-alive
                yield "event: ping\ndata: {}\n\n"

    return Response(gen(), mimetype="text/event-stream")



if __name__ == "__main__":
    # Run insider docker on 0.0.0.0
    app.run(host="0.0.0.0", port=int(os.getenv("TRAIN_SERVICE_PORT", "7000")), debug=False, threaded=True)