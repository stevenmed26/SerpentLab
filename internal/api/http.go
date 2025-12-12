package api

import (
	"log"
	"math/rand"
	"net/http"
	"time"
	"path/filepath"
	"os"
	"bytes"
	"encoding/json"
	"fmt"
	"io"

	"github.com/gorilla/websocket"

	"github.com/stevenmed26/serpentlab/internal/game"
)

type frame struct {
	Tick   int     `json:"tick"`
	Width  int     `json:"width"`
	Height int     `json:"height"`
	Grid   []int32 `json:"grid"`
	Score  int     `json:"score"`
	Done   bool    `json:"done"`

	HeadX int      `json:"headX"`
	HeadY int      `json:"headY"`
}

// Allow any origin for dev. Tighten this in production.
var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

var httpClient = &http.Client{
	Timeout: 200 * time.Millisecond,
}

func policyServerURL() string {
	url := os.Getenv("POLICY_SERVER_URL")
	if url == "" {
		url = "http://localhost:6000/act"
	}
	return url
}

func (g *Game) RenderWidth() int {
    if g.cfg.WithWalls { return g.width + 2 }
    return g.width
}
func (g *Game) RenderHeight() int {
    if g.cfg.WithWalls { return g.height + 2 }
    return g.height
}


func getPolicyAction(grid []int32, width, height int) (game.Direction, error) {
	reqBody := struct {
		Grid []int32 `json:"grid"`
		Width int    `json:"width"`
		Height int   `json:"height"`
	} {
		Grid:   grid,
		Width:  width,
		Height: height,
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return 0, err
	}

	resp, err := httpClient.Post(policyServerURL(), "application/json", bytes.NewReader(data))
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		log.Printf("policy server error: %s", string(body))
		return 0, fmt.Errorf("policy server status: %s", resp.Status)
	}

	var res struct {
		Action int `json:"action"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return 0, err
	}

	// Map int -> Direction
	switch res.Action {
	case 0:
		return game.DirUp, nil
	case 1:
		return game.DirRight, nil
	case 2:
		return game.DirDown, nil
	case 3:
		return game.DirLeft, nil
	default:
		return game.DirUp, nil
	}
}

// StartHTTPServer serves the static web UI and a WebSocket endpoint
// that streams live Snake frames.
func StartHTTPServer(addr string) error {

	cwd, err := os.Getwd()
	if err != nil {
		return err
	}
	rootDir := filepath.Dir(filepath.Dir(cwd))
	webDir := filepath.Join(rootDir, "web/src")

	log.Println("Serving static files from:", webDir)
	mux := http.NewServeMux()

	// Serve index.html explicitly on "/"
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			// For any other path (e.g. /app.js), let the file server handle it.
			http.FileServer(http.Dir(webDir)).ServeHTTP(w, r)
			return
		}
		http.ServeFile(w, r, filepath.Join(webDir, "index.html"))
	})

	// WebSocket endpoint for live game stream
	mux.HandleFunc("/ws/game", handleWSGame)

	log.Printf("HTTP/WebSocket server listening on %s", addr)
	return http.ListenAndServe(addr, mux)
}

func handleWSGame(w http.ResponseWriter, r *http.Request) {
    conn, err := upgrader.Upgrade(w, r, nil)
    if err != nil {
        log.Printf("websocket upgrade error: %v", err)
        return
    }
    defer conn.Close()

    mode := r.URL.Query().Get("mode")
    if mode == "" {
        mode = "random"
    }
    log.Printf("New WS connection, mode=%s", mode)

    cfg := game.Config{
        Width:     10,  // MUST match training config / checkpoint
        Height:    10,
        WithWalls: true,
		// MaxStepsWithoutFood: 80, // Don't force ant-stall on trained model
    }
    g := game.NewGame(cfg)
	lastDir := game.DirUp // Default direction

    ticker := time.NewTicker(100 * time.Millisecond) // ~10 FPS
    defer ticker.Stop()

    tick := 0

    sendFrame := func(done bool) error {
		head := g.Head()

        f := frame{
            Tick:   tick,
            Width:  g.Width(),
            Height: g.Height(),
            Grid:   g.Grid(),
            Score:  g.Score(),
            Done:   done,
			HeadX: head.X + (func() int { if g.cfg.WithWalls { return 1 }; return 0}),
			HeadY: head.Y + (func() int { if g.cfg.WithWalls { return 1 }; return 0}),
        }
        return conn.WriteJSON(f)
    }

    if err := sendFrame(false); err != nil {
        log.Printf("write initial frame error: %v", err)
        return
    }

	// --- Listen for reset message ---
	resetCh := make(chan struct{}, 1)
	manualActionCh := make(chan int, 1)

	go func() {
		for {
			_, data, err := conn.ReadMessage()
			if err != nil {
				log.Printf("ws read error: %v", err)
				return
			}

			var msg struct {
				Type string `json:"type"`
				Action int  `json:"action"`
			}
			if err := json.Unmarshal(data, &msg); err != nil {
				log.Printf("invalid client message: %v", err)
				continue
			}

			switch msg.Type {
			case"reset":
				select { case resetCh <- struct{}{}: default: }
			case "manual_action":
				select { case manualActionCh <- msg.Action: default: }	
			}
		}
	}()
    for range ticker.C {
        tick++
		
		select {
		case <-resetCh:
			log.Printf("reset requested by client")
			g.Reset()
			tick = 0
			if err := sendFrame(false); err != nil {
				log.Printf("write frame error after reset: %v", err)
				return
			}
			// Skip action
			continue
		default:
			// no reset
		}
        var actionDir game.Direction

        switch mode {
		case "manual":
			select {
			case a := <-manualActionCh:
				actionDir = game.Direction(a)
				lastDir = actionDir
			default:
				// no input - move straight
				actionDir = lastDir
			}
        case "policy":
            grid := g.Grid()
            a, err := getPolicyAction(grid, g.Width(), g.Height())
            if err != nil {
                log.Printf("policy error, falling back to random: %v", err)
                actionDir = game.Direction(rand.Intn(4))
            } else {
                actionDir = a
            }
        default: // "random" or anything else
            actionDir = game.Direction(rand.Intn(4))
        }

        _, done := g.Step(actionDir)

        if err := sendFrame(done); err != nil {
            log.Printf("write frame error: %v", err)
            return
        }

        if done {
			log.Printf("Resetting due to Death Cause: %v after foods: %v ", g.DeathCause(), g.Score())
            g.Reset()
            tick = 0
        }
    }
}
