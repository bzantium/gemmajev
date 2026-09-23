"""A local-view navigation interface with explicit, bounded exploration memory."""

from collections import Counter

DIRECTIONS = {"north": (-1, 0), "east": (0, 1), "south": (1, 0), "west": (0, -1)}
QUESTION = {
    "type": "choice",
    "instructions": (
        "Choose the next movement direction toward the goal from the available directions. "
        "Explore an unvisited passage; backtrack when the current branch is exhausted."
    ),
    "criteria": {name: f"Move one cell {name}." for name in DIRECTIONS},
}


def destination(position, action):
    dr, dc = DIRECTIONS[action]
    return position[0] + dr, position[1] + dc


class NavigationMemory:
    """Only public coordinates, local observations and actual feedback enter here.

    A DFS stack handles exhausted branches. At a junction, the model ranks the
    available unvisited passages. No hidden map or teacher route is stored.
    """

    def __init__(self, position, goal, size):
        self.position, self.goal, self.size = tuple(position), tuple(goal), size
        self.stack = [self.position]
        self.visits = Counter({self.position: 1})
        self.attempts = Counter()
        self.outcomes = {}
        self.previous = "none"
        self.cells = {}
        self.window = []

    def observe(self, local_state):
        lines = local_state.split("Local map:\n", 1)[1].splitlines()[:5]
        if len(lines) != 5 or any(len(line) != 5 for line in lines):
            raise ValueError("Expected a 5 by 5 local window")
        for r, line in enumerate(lines):
            for c, char in enumerate(line):
                if char not in "#.AXG":
                    raise ValueError(f"Unknown map symbol: {char}")
                coordinate = (self.position[0] + r - 2, self.position[1] + c - 2)
                self.cells[coordinate] = char in ".AG"
        self.window = lines

    def available(self):
        unvisited = [
            a
            for a in DIRECTIONS
            if self.cells.get(destination(self.position, a), False)
            and not self.visits[destination(self.position, a)]
        ]
        if unvisited:
            return unvisited
        if len(self.stack) > 1:
            return [a for a in DIRECTIONS if destination(self.position, a) == self.stack[-2]]
        return []

    def request(self, identifier):
        available = self.available()
        lines = [
            f"Position: row {self.position[0]}, column {self.position[1]}.",
            f"Goal offset: {self.goal[0] - self.position[0]} rows south, "
            f"{self.goal[1] - self.position[1]} columns east.",
            f"Previous move: {self.previous}. Current visits: {self.visits[self.position]}.",
            "Local map (# wall, . open, A agent, X outside):",
            *self.window,
        ]
        for action in DIRECTIONS:
            target = destination(self.position, action)
            edge = (self.position, target)
            lines.append(
                f"{action}: {'open' if self.cells.get(target, False) else 'blocked'}; "
                f"visits={self.visits[target]}; attempts={self.attempts[edge]}; "
                f"last={self.outcomes.get(edge, 'untried')}."
            )
        lines.append("Available directions: " + ", ".join(available) + ".")
        return {"id": identifier, "state": "\n".join(lines), "questions": {"move": QUESTION}}

    def transition(self, action, next_position, collision):
        old = self.position
        target = destination(old, action)
        self.attempts[(old, target)] += 1
        self.outcomes[(old, target)] = "blocked" if collision else "open"
        if collision:
            if tuple(next_position) != old:
                raise ValueError("Collision changed position")
            self.cells[target] = False
            return
        if tuple(next_position) != target:
            raise ValueError("Unexpected movement")
        if len(self.stack) > 1 and target == self.stack[-2]:
            self.stack.pop()
        else:
            if self.visits[target]:
                raise ValueError("Only the parent can be revisited under this policy")
            self.stack.append(target)
        self.position = target
        self.visits[target] += 1
        self.previous = action


def select_action(probabilities, available):
    """Select only a legal unexplored passage, or the required backtracking step."""
    if not available:
        return None
    return max(available, key=lambda action: probabilities[action])
