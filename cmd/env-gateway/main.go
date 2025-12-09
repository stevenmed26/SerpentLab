// cmd/env-gateway/main.go
package main

import (
	"log"
	"net"

	"google.golang.org/grpc"

	"github.com/stevenmed26/serpentlab/internal/grpcapi"
)

func main() {
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
