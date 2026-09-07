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


# import random 

# import matplotlib.pyplot as plt 

# import numpy as np 

 

# # 1. Define the landscape (our hill equation) 

# def get_height(x): 

#     return -x**2 + 10 

 

# # 2. Hill Climbing Algorithm Parameters 

# current_x = random.uniform(-5, 5)   # Pick a random start position between -5 and 5 

# step_size = 0.1                     # Size of steps to take left or right 

# history_x = [current_x]             # List to track steps for the final plot 

# history_y = [get_height(current_x)]  # List to track heights for the final plot 

 

# # 3. Core Search Loop 

# while True: 

#     current_height = get_height(current_x) 

     

#     # Evaluate immediate neighbor steps 

#     next_x_left = current_x - step_size 

#     next_x_right = current_x + step_size 

     

#     height_left = get_height(next_x_left) 

#     height_right = get_height(next_x_right) 

#     # Move right if it goes higher 

#     if height_right > current_height and height_right >= height_left: 

#         current_x = next_x_right 

#     # Move left if it goes higher 

#     elif height_left > current_height and height_left > height_right: 

#         current_x = next_x_left 

#     # Stop if both directions go downward 

#     else: 

#         break 

         

#     # Log the step to draw the path later 

#     history_x.append(current_x) 

#     history_y.append(get_height(current_x)) 

 

# print(f"✅ Algorithm finished! Peak found at x = {current_x:.2f}") 

 

# # 4. Generate the Visual Graph 

# # Create 500 smooth data points along the x-axis to draw the curve of the hill 

# x_axis = np.linspace(-6, 6, 500) 

# y_axis = get_height(x_axis) 

 

# # Initialize the plot size 

# plt.figure(figsize=(10, 6)) 

 

# # Plot the smooth green line representing the hill terrain 

# plt.plot(x_axis, y_axis, color='green', label='The Hill Terrain (y = -x² + 10)', linewidth=2) 

# # Plot the path your algorithm walked as dotted blue lines 

# plt.plot(history_x, history_y, color='blue', linestyle=':', label='Path Walked by Algorithm') 

 

# # Mark the starting position with an orange dot 

# plt.scatter(history_x[0], history_y[0], color='orange', s=150, zorder=5, label=f'Start Position (x={history_x[0]:.2f})') 

 

# # Mark the final peak with a prominent red dot 


# plt.scatter(current_x, get_height(current_x), color='red', s=200, marker='*', zorder=6, label=f'Peak Found (x={current_x:.2f})') 

 

# # Add labels, grid lines, and titles for readability 

# plt.title('Hill Climbing Algorithm Visualization', fontsize=14, fontweight='bold') 

# plt.xlabel('State Space (X Position)', fontsize=12) 

# plt.ylabel('Objective Function (Elevation / Height)', fontsize=11) 

# plt.axhline(0, color='black', linewidth=0.5) 

# plt.axvline(0, color='black', linewidth=0.5) 

# plt.grid(True, linestyle='--', alpha=0.6) 

# plt.legend(loc='lower center', fontsize=10) 

# # Save the chart to a file when running in a non-interactive environment
# output_file = 'hill_climbing.png'
# plt.tight_layout()
# plt.savefig(output_file)
# print(f'Plot saved to {output_file}')