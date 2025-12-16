// internal/api/train_proxy.go

package api

import (
	"net/http/httputil"
	"net/url"
	"net/http"
	"os"
)

func registerTrainerRoutes(mux *http.ServeMux) {
	trainerURL := os.Getenv("TRAINER_SERVICE_URL")
	if trainerURL == "" {
		trainerURL = "http://trainer-service:7000"
	}

	u, err := url.Parse(trainerURL)
	if err != nil {
		// If fail, panic
		panic(err)
	}

	proxy := httputil.NewSingleHostReverseProxy(u)

	mux.HandleFunc("/api/train/start", func(w http.ResponseWriter, r *http.Request) {
		r.URL.Path = "/start"
		proxy.ServeHTTP(w, r)
	})
	mux.HandleFunc("/api/train/stop", func(w http.ResponseWriter, r *http.Request) {
		r.URL.Path = "/stop"
		proxy.ServeHTTP(w, r)
	})
	mux.HandleFunc("/api/train/status", func(w http.ResponseWriter, r *http.Request) {
		r.URL.Path = "/status"
		proxy.ServeHTTP(w, r)
	})
	mux.HandleFunc("/api/train/metrics", func(w http.ResponseWriter, r *http.Request) {
		r.URL.Path = "/metrics"
		proxy.ServeHTTP(w, r)
	})
}

