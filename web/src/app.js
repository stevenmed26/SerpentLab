(function () {
  const statusEl = document.getElementById("status");
  const infoEl = document.getElementById("info");
  /** @type {HTMLCanvasElement} */
  const canvas = document.getElementById("board");
  const ctx = canvas.getContext("2d");
  const modeSelect = document.getElementById("mode");

  let currentWs = null;

  function connect() {
    if (currentWs) {
      currentWs.close();
      currentWs = null;
    }

    const mode = modeSelect.value || "random";
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const wsUrl = `${protocol}://${window.location.host}/ws/game?mode=${encodeURIComponent(mode)}`;

    statusEl.textContent = `Connecting (${mode}) to ${wsUrl}...`;
    const ws = new WebSocket(wsUrl);
    currentWs = ws;

    ws.onopen = () => {
      statusEl.textContent = `Connected (${mode}).`;
    };

    ws.onclose = () => {
      if (currentWs === ws) {
        statusEl.textContent = `Disconnected (${mode}). Reconnecting in 2 seconds...`;
        setTimeout(connect, 2000);
      }
    };

    ws.onerror = (err) => {
      console.error("WebSocket error:", err);
      statusEl.textContent = "WebSocket error. Check console.";
    };

    ws.onmessage = (event) => {
      const frame = JSON.parse(event.data);
      renderFrame(frame);
    };
  }
  const resetBtn = document.getElementById("reset-btn");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      console.log("Reset button clicked");
      if (currentWs && currentWs.readyState === WebSocket.OPEN) {
        console.log("Sending reset over WebSocket");
        currentWs.send(JSON.stringify({ type: "reset" }))
      } else {
        console.warn("WebSocket not open, reloading");
        // reload as fallback
        window.location.reload();
      }
    });
  }

  function renderFrame(frame) {
    const { width, height, grid, score, tick, done } = frame;

    const cellW = canvas.width / width;
    const cellH = canvas.height / height;

    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        const val = grid[idx];
        if (val === 0) continue;

        // 0 = empty, 1 = snake, 2 = food, 3 = wall
        switch (val) {
          case 1:
            ctx.fillStyle = "#00ff66"; // snake
            break;
          case 2:
            ctx.fillStyle = "#ff3333"; // food
            break;
          case 3:
            ctx.fillStyle = "#555555"; // wall
            break;
          default:
            ctx.fillStyle = "#ffffff";
        }

        ctx.fillRect(x * cellW, y * cellH, cellW, cellH);
      }
    }

    infoEl.textContent = `Score: ${score} | Tick: ${tick} | ${done ? "Episode done (auto-reset)" : "Running"}`;
  }

  // Reconnect with new mode when user changes dropdown
  modeSelect.addEventListener("change", () => {
    connect();
  });

  // Initial connect
  connect();
})();
