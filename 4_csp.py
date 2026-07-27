"""
Program 4: Constraint Satisfaction Problem (Map Coloring using Backtracking)
Time Complexity: O(d^n) where d = domain size, n = variables (worst case)
Space Complexity: O(n)
"""


def is_consistent(variable, color, assignment, constraints):
    """Check if assigning color to variable is consistent with constraints."""
    for neighbor in constraints.get(variable, []):
        if neighbor in assignment and assignment[neighbor] == color:
            return False
    return True


def backtrack(variables, domains, constraints, assignment):
    """Solve CSP using backtracking with forward checking."""
    if len(assignment) == len(variables):
        return assignment

    # Select unassigned variable (MRV heuristic)
    unassigned = [v for v in variables if v not in assignment]
    variable = min(unassigned, key=lambda v: sum(
        1 for color in domains[v]
        if is_consistent(v, color, assignment, constraints)
    ))

    for color in domains[variable]:
        if is_consistent(variable, color, assignment, constraints):
            assignment[variable] = color
            result = backtrack(variables, domains, constraints, assignment)
            if result is not None:
                return result
            del assignment[variable]

    return None


def solve_csp(variables, domains, constraints):
    """Solve the CSP problem."""
    return backtrack(variables, domains, constraints, {})


# Driver Code
if __name__ == "__main__":
    # Map Coloring Problem (Australia)
    variables = ['WA', 'NT', 'SA', 'QLD', 'NSW', 'VIC', 'TAS']

    colors = ['Red', 'Green', 'Blue']
    domains = {v: colors[:] for v in variables}

    # Adjacency constraints (neighbors cannot have same color)
    constraints = {
        'WA': ['NT', 'SA'],
        'NT': ['WA', 'SA', 'QLD'],
        'SA': ['WA', 'NT', 'QLD', 'NSW', 'VIC'],
        'QLD': ['NT', 'SA', 'NSW'],
        'NSW': ['QLD', 'SA', 'VIC'],
        'VIC': ['SA', 'NSW', 'TAS'],
        'TAS': ['VIC']
    }

    print("Map Coloring Problem (Australia)")
    print("Colors available:", colors)
    print()

    solution = solve_csp(variables, domains, constraints)
    if solution:
        print("Solution found:")
        for region, color in solution.items():
            print(f"  {region:4s} -> {color}")
    else:
        print("No solution exists.")
