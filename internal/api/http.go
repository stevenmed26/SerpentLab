// internal/api/http.go
package api

import (
	"log"
	"net/http"
	"os"
	"time"
	"path/filepath"
	"github.com/gorilla/websocket"
)

// Allow any origin for dev. Tighten this in production.
var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

var httpClient = &http.Client{
	Timeout: 200 * time.Millisecond,
}

// StartHTTPServer serves the static web UI and a WebSocket endpoint
// that streams live Snake frames.
func StartHTTPServer(addr string) error {

	webDir, err := resolveWebDir()
	if err != nil {return err }

	mux := http.NewServeMux()

	// Serve index.html explicitly on "/"
	registerStaticRoutes(mux, webDir) // server
	registerViewerRoutes(mux) // game viewer
	registerTrainerRoutes(mux) // trainer proxy
	//Add more as needed

	log.Printf("HTTP/WebSocket server listening on %s", addr)
	return http.ListenAndServe(addr, mux)
}

func resolveWebDir() (string, error) {
	cwd, err := os.Getwd()
	if err != nil { return "", err }
	rootDir := filepath.Dir(filepath.Dir(cwd))
	return filepath.Join(rootDir, "web/src"), nil
}

