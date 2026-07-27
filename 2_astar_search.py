"""
Program 2: Informed Search (A* Algorithm)
Time Complexity: O(E log V) with binary heap
Space Complexity: O(V)
"""

import heapq


def astar(graph, start, goal, heuristic):
    """
    A* Search Algorithm.
    graph: dict of {node: [(neighbor, cost), ...]}
    heuristic: dict of {node: estimated_cost_to_goal}
    """
    # Priority queue: (f_score, node, path, g_score)
    open_set = [(heuristic[start], start, [start], 0)]
    visited = set()

    while open_set:
        f, current, path, g = heapq.heappop(open_set)

        if current == goal:
            return path, g

        if current in visited:
            continue
        visited.add(current)

        for neighbor, cost in graph[current]:
            if neighbor not in visited:
                new_g = g + cost
                new_f = new_g + heuristic[neighbor]
                heapq.heappush(open_set, (new_f, neighbor, path + [neighbor], new_g))

    return None, float('inf')


# Driver Code
if __name__ == "__main__":
    # Weighted graph: node -> [(neighbor, edge_cost), ...]
    graph = {
        'A': [('B', 1), ('C', 4)],
        'B': [('D', 5), ('E', 12)],
        'C': [('E', 8)],
        'D': [('F', 2)],
        'E': [('F', 3)],
        'F': []
    }

    # Heuristic values (estimated distance to goal 'F')
    heuristic = {
        'A': 10,
        'B': 8,
        'C': 5,
        'D': 3,
        'E': 3,
        'F': 0
    }

    path, cost = astar(graph, 'A', 'F', heuristic)
    print(f"A* Path: {' -> '.join(path)}")
    print(f"Total Cost: {cost}")
