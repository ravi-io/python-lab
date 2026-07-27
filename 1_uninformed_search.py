"""
Program 1: Uninformed Search (BFS and DFS)
Time Complexity: O(V + E) where V = vertices, E = edges
Space Complexity: O(V)
"""

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


def bfs(graph, start):
    """Breadth First Search - explores level by level."""
    visited = set([start])
    queue = deque([start])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return result


def dfs(graph, start):
    """Depth First Search - explores as deep as possible first."""
    visited = set()
    stack = [start]
    result = []

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            result.append(node)
            # Add neighbors in reverse to maintain left-to-right order
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)

    return result


# Driver Code
if __name__ == "__main__":
    print("Graph:", graph)
    print("BFS Traversal:", " -> ".join(bfs(graph, 'A')))
    print("DFS Traversal:", " -> ".join(dfs(graph, 'A')))
