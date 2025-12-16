// internal/game/static.go
package api

import (
	"net/http"
	"path/filepath"
)

func registerStaticRoutes(mux *http.ServeMux, webDir string) {
	fs := http.FileServer(http.Dir(webDir))

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" {
			http.ServeFile(w, r, filepath.Join(webDir, "index.html"))
			return
		}
		fs.ServeHTTP(w, r)
	})

	mux.HandleFunc("/train", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, filepath.Join(webDir, "train.html"))
	})
}

