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

# import heapq

# # 1. Define the Graph Structure (Adjacency List)
# # Format: { 'Current_Node': { 'Neighbor_Node': Actual_Edge_Cost, ... } }
# graph = {
#     'S': {'A': 1, 'B': 4},
#     'A': {'G': 6},
#     'B': {'G': 2},
#     'G': {}  # Goal node has no outgoing edges
# }

# # 2. Define the Admissible Heuristic Values h(n)
# # These match the exact estimation values to goal 'G' from the theory section
# heuristics = {
#     'S': 5,
#     'A': 3,
#     'B': 1,
#     'G': 0
# }

# def a_star_search(graph, start, goal, heuristics):
#     """
#     Executes the A* Search Algorithm to find the shortest path.
#     """
#     # Priority Queue elements format: (f_score, current_node)
#     # heapq automatically sorts by the first element in the tuple (f_score)
#     open_list = []
#     heapq.heappush(open_list, (heuristics[start], start))
    
#     # Track historical cost g(n) paid to reach each node
#     g_score = {node: float('inf') for node in graph}
#     g_score[start] = 0
    
#     # Dictionary to reconstruct the path once the goal is reached
#     parent_map = {}
    
#     print(f"--- Starting A* Search from {start} to {goal} ---")
    
#     while open_list:
#         # Pop the node with the lowest f_score
#         current_f, current_node = heapq.heappop(open_list)
#         print(f"\nExpanding Node: {current_node} (Current f-score: {current_f})")
        
#         # Goal test check
#         if current_node == goal:
#             print("-> Goal reached! Reconstructing path...")
#             return reconstruct_path(parent_map, start, goal), g_score[goal]
            
#         # Evaluate all neighbors of the current node
#         for neighbor, edge_weight in graph[current_node].items():
#             # Calculate tentative g(neighbor) = current paid cost + edge step cost
#             tentative_g = g_score[current_node] + edge_weight
            
#             # If this path to neighbor is better than any previous path
#             if tentative_g < g_score[neighbor]:
#                 g_score[neighbor] = tentative_g
#                 # f(n) = g(n) + h(n)
#                 f_score = tentative_g + heuristics[neighbor]
                
#                 parent_map[neighbor] = current_node
#                 heapq.heappush(open_list, (f_score, neighbor))
                
#                 print(f"  Calculated for Neighbor {neighbor}: "
#                       f"g={tentative_g}, h={heuristics[neighbor]} -> f={f_score}")
                
#     print("-> Failure: No path exists.")
#     return None, float('inf')

# def reconstruct_path(parent_map, start, goal):
#     """
#     Backtracks from goal to start using the parent map to construct the path.
#     """
#     path = []
#     current = goal
#     while current != start:
#         path.append(current)
#         current = parent_map[current]
#     path.append(start)
#     return path[::-1] # Reverse the list to get start -> goal order

# # 3. Execute the Algorithm
# optimal_path, total_cost = a_star_search(graph, 'S', 'G', heuristics)

# # 4. Print Final Results
# print("\n" + "="*40)
# print(f"✅ OPTIMAL PATH: {' -> '.join(optimal_path)}")
# print(f"✅ TOTAL MINIMIZED PATH COST: {total_cost}")
# print("="*40)