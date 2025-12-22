// internal/game/viewer_ws.go
package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"time"

	"github.com/stevenmed26/serpentlab/internal/game"
)

type frame struct {
	Tick   int     `json:"tick"`
	Width  int     `json:"width"`
	Height int     `json:"height"`
	Grid   []int32 `json:"grid"`
	Score  int     `json:"score"`
	Done   bool    `json:"done"`

	HeadX int `json:"headX"`
	HeadY int `json:"headY"`
}

func registerViewerRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/ws/game", handleWSGame)
}

func getPolicyAction(mode string, grid []int32, width, height int) (game.Direction, error) {
	reqBody := struct {
		Grid   []int32 `json:"grid"`
		Width  int     `json:"width"`
		Height int     `json:"height"`
	}{
		Grid:   grid,
		Width:  width,
		Height: height,
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return 0, err
	}

	resp, err := httpClient.Post(policyServerURL(mode), "application/json", bytes.NewReader(data))
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

func policyServerURL(mode string) string {
	var baseURL string
	switch mode {
	case "dqn":
		baseURL = os.Getenv("POLICY_SERVER_URL_DQN")
		return baseURL
	case "ppo":
		baseURL = os.Getenv("POLICY_SERVER_URL_PPO")
		return baseURL
	default:
		log.Printf("unsupported policy mode: %s - falling back to DQN", mode)
		baseURL = os.Getenv("POLICY_SERVER_URL_DQN")
		return baseURL
	}
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
		Width:     10, // MUST match training config / checkpoint
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

		grid, outW, outH, off := g.GridRender()

		f := frame{
			Tick:   tick,
			Width:  outW,
			Height: outH,
			Grid:   grid,
			Score:  g.Score(),
			Done:   done,
			HeadX:  head.X + off,
			HeadY:  head.Y + off,
		}
		return conn.WriteJSON(f)
	}

	if err := sendFrame(false); err != nil {
		log.Printf("write initial frame error: %v", err)
		return
	}

	// --- Listen for reset message ---
	paused := false
	pauseCh := make(chan bool, 1)
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
				Type   string `json:"type"`
				Action int    `json:"action"`
			}
			if err := json.Unmarshal(data, &msg); err != nil {
				log.Printf("invalid client message: %v", err)
				continue
			}

			switch msg.Type {
			case "reset":
				select {
				case resetCh <- struct{}{}:
				default:
				}
			case "manual_action":
				select {
				case manualActionCh <- msg.Action:
				default:
				}
			case "pause":
				select {
				case pauseCh <- true:
				default:
				}
			case "resume":
				select {
				case pauseCh <- false:
				default:
				}
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
		case p := <-pauseCh:
			paused = p
		default:
			// no reset
		}

		if paused {
			continue
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
		case "dqn", "ppo":
			obs := g.Grid()

			a, err := getPolicyAction(mode, obs, g.Width(), g.Height()) // Refine to DQN only
			if err != nil {
				log.Printf("%s policy error, falling back to random: %v", mode, err)
				actionDir = game.Direction(rand.Intn(4))
			} else {
				actionDir = a
			}
		default: // "random" or anything else
			actionDir = game.Direction(rand.Intn(4))
		}

		// _, done := g.Step(actionDir)
		done := g.Step(actionDir)

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
