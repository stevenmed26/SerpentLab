// cmd/env-gateway/main.go
package main

import (
	"log"
	"net"

	"google.golang.org/grpc"

	"github.com/stevenmed26/serpentlab/internal/api"
	"github.com/stevenmed26/serpentlab/internal/grpcapi"
)

func main() {
	// Start HTTP/Websocket server on :8080
	go func() {
		if err := api.StartHTTPServer(":8080"); err != nil {
			log.Fatalf("HTTP server error: %v", err)
		}
	}()

	// Start gRPC env server on :50051
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer()

	envServer := grpcapi.NewEnvServer()
	grpcapi.RegisterSnakeEnvServer(grpcServer, envServer)

	log.Println("env-gateway gRPC server listening on :50051")
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("failed to serve gRPC: %v", err)
	}
}
