# """
# Program 3: 8-Puzzle Problem (using A* with Manhattan Distance)
# Time Complexity: O(b^d) where b = branching factor, d = depth
# Space Complexity: O(b^d)
# """

# import heapq


# def manhattan_distance(state, goal):
#     """Calculate Manhattan distance heuristic."""
#     distance = 0
#     for i in range(9):
#         if state[i] != 0:
#             goal_idx = goal.index(state[i])
#             distance += abs(i // 3 - goal_idx // 3) + abs(i % 3 - goal_idx % 3)
#     return distance


# def get_neighbors(state):
#     """Get all possible states from current state."""
#     neighbors = []
#     blank = state.index(0)
#     row, col = blank // 3, blank % 3

#     moves = [(-1, 0, "Up"), (1, 0, "Down"), (0, -1, "Left"), (0, 1, "Right")]

#     for dr, dc, direction in moves:
#         new_row, new_col = row + dr, col + dc
#         if 0 <= new_row < 3 and 0 <= new_col < 3:
#             new_state = list(state)
#             new_idx = new_row * 3 + new_col
#             new_state[blank], new_state[new_idx] = new_state[new_idx], new_state[blank]
#             neighbors.append((tuple(new_state), direction))

#     return neighbors


# def solve_puzzle(start, goal):
#     """Solve 8-puzzle using A* search."""
#     start = tuple(start)
#     goal = tuple(goal)

#     if start == goal:
#         return []

#     # (f_score, counter, state, moves)
#     counter = 0
#     open_set = [(manhattan_distance(start, goal), counter, start, [])]
#     visited = set()

#     while open_set:
#         f, _, current, moves = heapq.heappop(open_set)

#         if current == goal:
#             return moves

#         if current in visited:
#             continue
#         visited.add(current)

#         for neighbor, direction in get_neighbors(current):
#             if neighbor not in visited:
#                 new_moves = moves + [direction]
#                 h = manhattan_distance(neighbor, goal)
#                 counter += 1
#                 heapq.heappush(open_set, (len(new_moves) + h, counter, neighbor, new_moves))

#     return None


# def print_state(state):
#     """Display the puzzle state."""
#     for i in range(0, 9, 3):
#         row = [str(x) if x != 0 else '_' for x in state[i:i+3]]
#         print(" ".join(row))
#     print()


# # Driver Code
# if __name__ == "__main__":
#     start = [1, 2, 3, 4, 0, 5, 6, 7, 8]
#     goal = [1, 2, 3, 4, 5, 6, 7, 8, 0]

#     print("Start State:")
#     print_state(start)
#     print("Goal State:")
#     print_state(goal)

#     solution = solve_puzzle(start, goal)
#     if solution:
#         print(f"Solution found in {len(solution)} moves:")
#         print(" -> ".join(solution))
#     else:
#         print("No solution exists.")


from collections import deque

def get_neighbors(state):
    neighbors = []
    zero_idx = state.index(0)
    r, c = divmod(zero_idx, 3)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for dr, dc in moves:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            neighbors = list(state)
            swap_idx = nr * 3 + nc
            neighbors[zero_idx], neighbors[swap_idx] = neighbors[swap_idx], neighbors[zero_idx]
            neighbors.append(tuple(neighbors))
    return neighbors

def solve_8_puzzle(start, goal):
    
  