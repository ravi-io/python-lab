from collections import deque

# Graph representation using adjacency list
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': [],
    'E': [],
    'F': [],
    'G': []
}

# Depth First Search (DFS)
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
        print("DFS Traversal:", end=" ")

    print(start, end=" ")
    visited.add(start)

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# Driver Code
dfs(graph, 'A')
print()