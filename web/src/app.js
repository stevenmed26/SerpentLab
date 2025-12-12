(function () {
  const statusEl = document.getElementById("status");
  const infoEl = document.getElementById("info");
  /** @type {HTMLCanvasElement} */
  const canvas = document.getElementById("board");
  const ctx = canvas.getContext("2d");

  const modeSelect = document.getElementById("mode");
  const pauseBtn = document.getElementById("pause-btn");
  const resumeBtn = document.getElementById("resume-btn");
  const resetBtn = document.getElementById("reset-btn");
  const speedSlider = document.getElementById("speed");
  const speedLabel = document.getElementById("speed-label");

  let currentWs = null;
  let paused = false;

  let lastRenderTime = 0;
  let lastFrameSwitchTime = null;
  let prevFrame = null;
  let currentFrame = null;

  // Control render smoothing
  let renderInterval = 100;
  speedLabel.textContent = `${renderInterval} ms`;

  speedSlider.addEventListener("input", () => {
    renderInterval = Number(speedSlider.value);
    speedLabel.textContent = `${renderInterval} ms`;
  });

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
      if (paused) return;

      const frame = JSON.parse(event.data);
      prevFrame = currentFrame;
      currentFrame = frame;
      lastFrameSwitchTime = performance.now();
    };
  }
  pauseBtn.addEventListener("click", () => {
    paused = true;
    statusEl.textContent = "Paused";
  });

  resumeBtn.addEventListener("click", () => {
    paused = false;
    statusEl.textContent = "Running";
  });

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

  modeSelect.addEventListener("change", () => {
    connect();
  })

  document.addEventListener("keydown", (e) => {
    if (modeSelect.value !== "manual") return;
    if (!currentWs || currentWs.readyState !== WebSocket.OPEN) return;

    let action = null;
    switch (e.key) {
      case "ArrowUp": action = 0; break;
      case "ArrowRight": action = 1; break;
      case "ArrowDown": action = 2; break;
      case "ArrowLeft": action = 3; break;
    }
    if (action !== null) {
      currentWs.send(JSON.stringify({ type: "manual_action", action }));
    }
  });

  function renderFrameInterpolated(prev, curr, alpha) {
    // If no previous frame
    if (!prev) {
      renderFrameBase(curr, null);
      return;
    }
    renderFrameBase(curr, {prev, alpha });
  }
  function renderFrame(frame, interp) {
    renderFrameBase(frame, interp);
  }

  function renderFrameBase(frame, interp) {
    const { width, height, grid, score, tick, done, headX, headY } = frame;

    const cellW = canvas.width / width;
    const cellH = canvas.height / height;

    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        const val = grid[idx];
        if (val === 0) continue;

        // 0 = empty, 1 = snake, 2 = food, 3 = wall, Don't color head
        if (val === 1 && x === headX && y === headY) {
          continue;
        }

        switch (val) {
          case 1:
            ctx.fillStyle = "#00aa44"; // snake
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

    //Interpolated head
    let drawHeadX = headX;
    let drawHeadY = headY;

    if (interp && interp.prev) {
      const prev = interp.prev;
      const alpha = interp.alpha;

      const prevHeadX = prev.headX ?? headX;
      const prevHeadY = prev.headY ?? headY;

      drawHeadX = prevHeadX + (headX - prevHeadX) * alpha;
      drawHeadY = prevHeadY + (headY - prevHeadY) * alpha
    }

    // Draw head overlay
    ctx.fillStyle = "#00ff66" // Head
    const headPx = drawHeadX * cellW;
    const headPy = drawHeadY * cellH;
    ctx.fillRect(headPx, headPy, cellW, cellH);

    infoEl.textContent = `Score: ${score} | Tick: ${tick} | ${done ? "Episode done (auto-reset)" : "Running"}`;
  }
  function renderLoop(timestamp) {
    if (!paused && currentFrame) {
      if (timestamp - lastRenderTime >= renderInterval) {
        let alpha = 1.0;
        if (prevFrame) {
          const dt = timestamp - lastFrameSwitchTime;
          alpha = Math.max(0, Math.min(1, dt / renderInterval));
        } else {
          alpha = 1.0;
        }
        renderFrameInterpolated(prevFrame, currentFrame, alpha);
        lastRenderTime = timestamp;
      }
    }
    requestAnimationFrame(renderLoop);
  }
  requestAnimationFrame(renderLoop);


  // Initial connect
  connect();
})();
