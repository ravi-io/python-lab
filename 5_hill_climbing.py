"""
Program 5: Hill Climbing Algorithm (Steepest Ascent)
Time Complexity: O(n * m) where n = iterations, m = neighbors per state
Space Complexity: O(1) - only stores current state
"""

import random


def objective_function(x):
    """Function to maximize: f(x) = -(x-3)^2 + 9 (peak at x=3, max=9)."""
    return -(x - 3) ** 2 + 9


def hill_climbing(func, start, step_size=0.1, max_iterations=1000):
    """
    Steepest Ascent Hill Climbing for maximization.
    Returns the best solution found and its value.
    """
    current = start
    current_value = func(current)

    for i in range(max_iterations):
        # Generate neighbors
        neighbors = [current + step_size, current - step_size]

        # Find the best neighbor
        best_neighbor = max(neighbors, key=func)
        best_value = func(best_neighbor)

        # If no improvement, we've reached a peak
        if best_value <= current_value:
            break

        current = best_neighbor
        current_value = best_value

    return current, current_value


def hill_climbing_nqueens(n=8, max_restarts=100):
    """
    Hill Climbing for N-Queens problem with random restarts.
    Minimizes the number of attacking pairs.
    """
    def count_attacks(state):
        attacks = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if state[i] == state[j] or abs(state[i] - state[j]) == j - i:
                    attacks += 1
        return attacks

    def get_best_neighbor(state):
        best_state = state[:]
        best_attacks = count_attacks(state)

        for col in range(n):
            for row in range(n):
                if state[col] != row:
                    new_state = state[:]
                    new_state[col] = row
                    attacks = count_attacks(new_state)
                    if attacks < best_attacks:
                        best_attacks = attacks
                        best_state = new_state[:]

        return best_state, best_attacks

    for restart in range(max_restarts):
        # Random initial state
        current = [random.randint(0, n - 1) for _ in range(n)]
        current_attacks = count_attacks(current)

        while current_attacks > 0:
            neighbor, neighbor_attacks = get_best_neighbor(current)
            if neighbor_attacks >= current_attacks:
                break  # Local minimum reached
            current = neighbor
            current_attacks = neighbor_attacks

        if current_attacks == 0:
            return current, restart + 1

    return None, max_restarts


# Driver Code
if __name__ == "__main__":
    # Simple function optimization
    print("=== Hill Climbing: Function Optimization ===")
    print("Maximizing f(x) = -(x-3)^2 + 9")
    start = random.uniform(-10, 10)
    solution, value = hill_climbing(objective_function, start)
    print(f"Starting point: {start:.4f}")
    print(f"Solution: x = {solution:.4f}, f(x) = {value:.4f}")
    print()

    # N-Queens Problem
    print("=== Hill Climbing: 8-Queens Problem ===")
    solution, restarts = hill_climbing_nqueens(8)
    if solution:
        print(f"Solution found in {restarts} restart(s): {solution}")
        print("Board:")
        for row in range(8):
            line = ""
            for col in range(8):
                line += "Q " if solution[col] == row else ". "
            print(f"  {line}")
    else:
        print("No solution found.")
