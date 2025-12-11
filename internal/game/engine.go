package game

// Cell represents the contents of a grid cell.
import (
	"math/rand"
	"time"
)

type Cell int8

const (
	CellEmpty Cell = iota
	CellSnake
	CellFood
	CellWall
)

// Direction is the movement direction of the snake.
type Direction int8

const (
	DirUp Direction = iota
	DirRight
	DirDown
	DirLeft
)

// Config holds configuration for a Snake game instance.
type Config struct {
	Width              int
	Height             int
	WithWalls          bool
	MaxStepsWithoutFood int // optional anti-stalling; 0 disables
}

// Point represents a coordinate on the board.
type Point struct {
	X int
	Y int
}

// Game holds the full state for a single Snake episode.
type Game struct {
	cfg Config

	width  int
	height int

	snake      []Point // head is snake[0]
	dir        Direction
	food       Point
	alive      bool
	score      int
	stepIndex  int
	stepsSinceFood int
	lastDeathCause string
}

// NewGame creates a new game with the given configuration.
func NewGame(cfg Config) *Game {
	if cfg.Width <= 0 {
		cfg.Width = 10
	}
	if cfg.Height <= 0 {
		cfg.Height = 10
	}
	g := &Game{
		cfg:    cfg,
		width:  cfg.Width,
		height: cfg.Height,
	}
	rand.Seed(time.Now().UnixNano())
	g.Reset()
	return g
}

// Reset starts a new episode with a fresh snake and food placement.
func (g *Game) Reset() {
	// Basic starting position: snake of length 1 in the center, facing right.
	g.snake = g.snake[:0]
	start := Point{X: g.width / 2, Y: g.height / 2}
	g.snake = append(g.snake, start)

	g.dir = DirRight
	g.placeFood()
	g.alive = true
	g.score = 0
	g.stepIndex = 0
	g.stepsSinceFood = 0
}

//abs helper
func abs(x int) int {
	if x < 0 { return -x}
	return x
}

// Step advances the game by one tick using the provided action.
//
// Returns (reward, done).
func (g *Game) Step(action Direction) (float64, bool) {
	if !g.alive {
		return 0.0, true
	}

	// Epsilon default per-step penalty
	reward := -0.02

	// Update direction - disallow direct reversal if you like.
	if isValidTurn(g.dir, action) {
		g.dir = action
	}

	// Compute new head position.
	head := g.snake[0]
	next := head
	switch g.dir {
	case DirUp:
		next.Y--
	case DirDown:
		next.Y++
	case DirLeft:
		next.X--
	case DirRight:
		next.X++
	}

	oldDist := abs(head.X - g.food.X) + abs(head.Y - g.food.Y)
	newDist := abs(next.X - g.food.X) + abs(next.Y - g.food.Y)

	if newDist < oldDist {
		reward += 0.15 // moved closer
	} else if newDist > oldDist {
		reward -= 0.15 // moved farther
	} else {
		// Same distance, slightly bad
		reward -= 0.02
	}

	// Check collisions with walls.
	if g.cfg.WithWalls {
		if next.X < 0 || next.X >= g.width || next.Y < 0 || next.Y >= g.height {
			g.alive = false
			g.lastDeathCause = "wall"
			reward = -15.0
			return reward, true
		}
	} else {
		// Wrap-around mode if !WithWalls.
		if next.X < 0 {
			next.X = g.width - 1
		} else if next.X >= g.width {
			next.X = 0
		}
		if next.Y < 0 {
			next.Y = g.height - 1
		} else if next.Y >= g.height {
			next.Y = 0
		}
	}

	// Check collision with self.
	for _, p := range g.snake {
		if p == next {
			g.alive = false
			g.lastDeathCause = "self"
			reward = -12.0
			return reward, true
		}
	}

	// Move snake: add new head.
	g.snake = append([]Point{next}, g.snake...)

	// Check if we ate food.
	if next == g.food {
		g.score++

		// Increasing food reward:
		foodsEaten := g.score
		foodReward := 6.0 + 0.5*float64(foodsEaten)
		reward += foodReward
		g.stepsSinceFood = 0
		g.placeFood()
	} else {
		// No food: remove tail.
		g.snake = g.snake[:len(g.snake)-1]
		g.stepsSinceFood++

		// --- Hunger penalty after grace period ---
		const hungerGrace = 10 // free steps after food
		const hungerScale = 0.02 // per-step penalty
		const hungerMaxPenalty = 0.4 // cap

		if g.stepsSinceFood > hungerGrace {
			hunger := float64(g.stepsSinceFood - hungerGrace)
			extra := hunger * hungerScale
			if extra > hungerMaxPenalty {
				extra = hungerMaxPenalty
			}
			reward -= extra
		}
	}

	g.stepIndex++

	// Optional anti-stall condition.
	if g.cfg.MaxStepsWithoutFood > 0 && g.stepsSinceFood >= g.cfg.MaxStepsWithoutFood {
		g.alive = false
		g.lastDeathCause = "stall"
		reward = -8.0
		return reward, true
	}

	return reward, !g.alive
}

// Width returns the board width.
func (g *Game) Width() int { return g.width }

// Height returns the board height.
func (g *Game) Height() int { return g.height }

// Score returns the current score (e.g. apples eaten).
func (g *Game) Score() int { return g.score }

// StepIndex returns the number of steps taken in this episode.
func (g *Game) StepIndex() int { return g.stepIndex }

// Getter function for death cause
func (g *Game) DeathCause() string {return g.lastDeathCause}

// Grid returns a flattened representation of the board as []int32, matching the proto.
// 0 = empty, 1 = snake, 2 = food, 3 = wall.
func (g *Game) Grid() []int32 {
	grid := make([]int32, g.width*g.height)

	// Walls
	if g.cfg.WithWalls {
		for x := 0; x < g.width; x++ {
			grid[index(x, 0, g.width)] = int32(CellWall)
			grid[index(x, g.height-1, g.width)] = int32(CellWall)
		}
		for y := 0; y < g.height; y++ {
			grid[index(0, y, g.width)] = int32(CellWall)
			grid[index(g.width-1, y, g.width)] = int32(CellWall)
		}
	}

	// Snake
	for _, p := range g.snake {
		if inBounds(p.X, p.Y, g.width, g.height) {
			grid[index(p.X, p.Y, g.width)] = int32(CellSnake)
		}
	}

	// Food
	if inBounds(g.food.X, g.food.Y, g.width, g.height) {
		grid[index(g.food.X, g.food.Y, g.width)] = int32(CellFood)
	}

	return grid
}

// --- helpers ---

func isValidTurn(oldDir, newDir Direction) bool {
	// Disallow 180-degree turns to avoid weird self-collisions.
	switch oldDir {
	case DirUp:
		return newDir == DirLeft || newDir == DirRight || newDir == DirUp
	case DirDown:
		return newDir == DirLeft || newDir == DirRight || newDir == DirDown
	case DirLeft:
		return newDir == DirUp || newDir == DirDown || newDir == DirLeft
	case DirRight:
		return newDir == DirUp || newDir == DirDown || newDir == DirRight
	default:
		return true
	}
}

func (g *Game) placeFood() {
	// Collect all possible empty cells
	type cell struct{ X, Y int }
	var candidates []cell

	for y := 0; y < g.height; y++ {
		for x := 0; x < g.height; x++ {

			// Skip walls if enabled (outer border)
			if g.cfg.WithWalls &&
				(x == 0 || x == g.width-1 || y == 0 || y == g.height-1) {
				continue
			}

			// Skip snake body
			occupied := false
			for _, seg := range g.snake {
				if seg.X == x && seg.Y == y {
					occupied = true
					break
				}
			}
			if occupied {
				continue
			}

			candidates = append(candidates, cell{X: x, Y: y})
		}
	}

	if len(candidates) == 0 {
		// No free cells? Just keep existing food
		return
	}

	// Pick candidate
	c := candidates[rand.Intn(len(candidates))]
	g.food = Point{X: c.X, Y: c.Y}
}

func index(x, y, width int) int {
	return y*width + x
}

func inBounds(x, y, width, height int) bool {
	return x >= 0 && x < width && y >= 0 && y < height
}
