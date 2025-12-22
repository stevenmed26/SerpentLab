// internal/grpcapi/server.go
package grpcapi

import (
	"context"
	"fmt"
	"sync"

	"github.com/stevenmed26/serpentlab/internal/game"
)

// EnvServer implements the SnakeEnv gRPC service.
// The generated interface types (SnakeEnvServer, ResetRequest, etc.) live
// in this same package because of the go_package option in the proto.
type EnvServer struct {
	UnimplementedSnakeEnvServer

	mu       sync.Mutex
	sessions map[string]*game.Game
	nextID   int64
}

// NewEnvServer creates a new EnvServer instance.
func NewEnvServer() *EnvServer {
	return &EnvServer{
		sessions: make(map[string]*game.Game),
	}
}

// Reset starts a new episode (or restarts an existing one).
func (s *EnvServer) Reset(ctx context.Context, req *ResetRequest) (*ResetResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	sessionID := req.GetSessionId()
	if sessionID == "" {
		s.nextID++
		sessionID = fmt.Sprintf("session-%d", s.nextID)
	}

	cfg := game.Config{
		Width:     int(req.GetWidth()),
		Height:    int(req.GetHeight()),
		WithWalls: req.GetWithWalls(),
		// You can tune this later; 0 disables anti-stall.
		MaxStepsWithoutFood: 80,
	}

	g := game.NewGame(cfg)
	g.Reset()

	s.sessions[sessionID] = g

	resp := &ResetResponse{
		SessionId: sessionID,
		Grid:      g.Grid(),
		Width:     int32(g.Width()),
		Height:    int32(g.Height()),
		Score:     int32(g.Score()),
		Done:      false,
	}

	return resp, nil
}

// Step advances the given session by one action.
func (s *EnvServer) Step(ctx context.Context, req *StepRequest) (*StepResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	sessionID := req.GetSessionId()
	g, ok := s.sessions[sessionID]
	if !ok {
		return nil, fmt.Errorf("unknown session_id: %s", sessionID)
	}

	action := toDirection(req.GetAction())
	//reward, done := g.Step(action)
	done := g.Step(action)

	resp := &StepResponse{
		Grid:   g.Grid(),
		Width:  int32(g.Width()),
		Height: int32(g.Height()),
		// Reward:    float32(reward),
		Done:           done,
		Score:          int32(g.Score()),
		StepIndex:      int32(g.StepIndex()),
		DeathCause:     toProtoDeathCause(g.DeathCause()),
		DeltaDist:      string(g.DeltaDist()),
		AteFood:        bool(g.AteFood()),
		StepsSinceFood: int32(g.StepsSinceFood()),
	}

	if done {
		// You can either keep finished sessions for replay or delete them.
		// For now, keep them to allow the trainer to inspect final state.
	}

	return resp, nil
}

func toDirection(a int32) game.Direction {
	switch a {
	case 0:
		return game.DirUp
	case 1:
		return game.DirRight
	case 2:
		return game.DirDown
	case 3:
		return game.DirLeft
	default:
		// Clamp invalid actions to "no-op" (keep current dir),
		// but Game.Step will handle validity anyway.
		return game.DirUp
	}
}

func toProtoDeathCause(c game.DeathCause) DeathCause {
	switch c {
	case game.DeathCause_DEATH_CAUSE_WALL:
		return DeathCause_DEATH_CAUSE_WALL
	case game.DeathCause_DEATH_CAUSE_SELF:
		return DeathCause_DEATH_CAUSE_SELF
	case game.DeathCause_DEATH_CAUSE_STALL:
		return DeathCause_DEATH_CAUSE_STALL
	default:
		return DeathCause_DEATH_CAUSE_UNSPECIFIED
	}
}
