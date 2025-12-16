const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");
const trainBtn = document.getElementById("train-btn");
let training = false;

function val(id) {return document.getElementById(id).value; }

async function refreshStatus() {
    const res = await fetch("/api/train/status");
    const j = await res.json();
    
    if (j.ok && j.status) {
        const s = j.status;
        statusEl.textContent = 
            `Status: ${s.running ? "Running" : "Stopped"}\n` +
            `Episode: ${s.episode}\n` +
            `Last Reward: ${(s.last_reward ?? 0).toFixed(2)}\n` +
            `Avg Last 50: ${(s.avg_last_50 ?? 0).toFixed(2)}\n` +
            `Foods: ${s.foods}\n` +
            `Message: ${s.message}`;
    } else {
        statusEl.textContent = "Error fetching status";
    }

    if (j.status?.running !== undefined) {
        training = j.status.running;
        trainBtn.textContent = training ? "Stop" : "Start";
        trainBtn.classList.toggle("btn-danger", training);
    }
}

const viewerBtn = document.getElementById("viewer-btn");

if (viewerBtn) {
viewerBtn.addEventListener("click", () => {
    console.log("Navigating to viewer page");
    window.location.assign("/");
});
}

document.getElementById("clear-log")?.addEventListener("click", () => {
  logEl.textContent = "";
});


trainBtn.addEventListener("click", async () => {
    if (!training) {
        // START
        const body = {
            width: Number(val("width")),
            height: Number(val("height")),
            num_episodes: Number(val("episodes")),
            batch_size: Number(val("batch")),
            buffer_capacity: Number(val("buffer")),
            eps_end: Number(val("epsEnd")),
            eps_decay_episodes: Number(val("epsDecay")),
            checkpoint_interval: Number(val("ckpt")),
            with_walls: true,
        };

        const res = await fetch("/api/train/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });

        logEl.textContent += `\nSTART: ${res.status}` + await res.text() + "\n";
        training = true;
        trainBtn.textContent = "Stop";
        trainBtn.classList.add("btn-danger");
    } else {
        // STOP
        const res = await fetch("/api/train/stop", { method: "POST" });

        logEl.textContent += `\nSent stop request to server: ${res.status}` + await res.text() + "\n";
        training = false;
        trainBtn.textContent = "Start";
        trainBtn.classList.remove("btn-danger");
    }

    refreshStatus();
});

// SSE metrics
const es = new EventSource("/api/train/metrics");
es.onmessage = (e) => {
    try {
        const m = JSON.parse(e.data);
        if (m.episode % 50 === 0) {
            logEl.textContent =
             `[ep ${m.episode}] eps=${m.eps.toFixed(3)} reward=${m.reward.toFixed(2)} foods=${m.foods} avg50=${m.avg_last_50.toFixed(2)}\n` +
             logEl.textContent.split("\n").slice(0, 40).join("\n");

            logEl.parentElement.scrollTop = logEl.parentElement.scrollHeight;
        }
    } catch {}
};

setInterval(refreshStatus, 5000);
refreshStatus();