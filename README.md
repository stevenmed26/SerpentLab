# SerpentLab | Hybrid Go + Python Snake RL Lab

SerpentLab is a containerized experimentation lab where multiple Snake game agents
learn via reinforcement learning, compete in tournaments, and stream their games
live to a web dashboard.

- **Go** powers the game engine, gRPC environment service, HTTP API, and WebSocket
  live viewer.
- **Python (PyTorch)** implements the RL agent and training loop, treating the Go
  service as a remote environment.

---

## Features

- 🐍 **Deterministic Snake game engine** with configurable board size, speed, and obstacles.
- 🧠 **Reinforcement learning trainer (Python + PyTorch)** communicating with Go via gRPC.
- 📊 **Live dashboard** to watch games in real-time and inspect metrics.
- 🧪 **Tournament mode** to compare different model checkpoints and track leaderboards.
- 📦 **Dockerized microservices** with a single `docker-compose up` for local deployment.

---

## Architecture

SerpentLab is split into three main components:

1. **env-gateway (Go)**
   - Implements the Snake game logic.
   - Exposes a `SnakeEnv` gRPC service consumed by the Python trainer.
   - Serves a REST API and WebSocket endpoints for the frontend.
   - Tracks game stats and aggregates metrics.

2. **rl-trainer (Python)**
   - Connects to the `SnakeEnv` gRPC service to run training episodes.
   - Implements DQN / policy gradient agents in PyTorch.
   - Periodically saves model checkpoints with performance metadata.

3. **web UI (HTML/JS)**
   - Connects to the WebSocket endpoint to render live games.
   - Displays leaderboards, model versions, and training metrics.

---

## Tech Stack

- **Backend (env-gateway):** Go, gRPC, HTTP, WebSockets
- **RL Trainer:** Python, PyTorch, gRPC client
- **Frontend:** HTML, JavaScript, WebSocket client
- **Containerization:** Docker, docker-compose
- **(Optional) Persistence:** Postgres or SQLite for models/leaderboards
- **(Optional) Observability:** Prometheus, Grafana

---

## Getting Started

### Prerequisites

- Go >= 1.22
- Python >= 3.11
- Docker + docker-compose
- `protoc` and the Go/Python gRPC plugins

### Quickstart (local dev)

```bash
# 1. Generate gRPC code
make proto

# 2. Run env-gateway (Go)
make run-env-gateway

# 3. Run Python trainer
cd python/trainer
python main.py

# 4. Run frontend (served by Go or static dev server)
# Visit: http://localhost:8080
