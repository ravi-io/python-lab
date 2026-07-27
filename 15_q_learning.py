"""
Program 15: Reinforcement Learning (Q-Learning)
Time Complexity: O(episodes * max_steps) per training
Space Complexity: O(S * A) for Q-table, S=states, A=actions
"""

import random


class GridWorld:
    """Simple grid world environment."""

    def __init__(self, rows=4, cols=4):
        self.rows = rows
        self.cols = cols
        self.start = (0, 0)
        self.goal = (rows - 1, cols - 1)
        # Obstacles
        self.obstacles = {(1, 1), (2, 2), (1, 3)}
        self.state = self.start

    def reset(self):
        """Reset to start state."""
        self.state = self.start
        return self.state

    def get_actions(self):
        """Available actions: 0=Up, 1=Down, 2=Left, 3=Right."""
        return [0, 1, 2, 3]

    def step(self, action):
        """Take action, return (next_state, reward, done)."""
        row, col = self.state
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            row = min(self.rows - 1, row + 1)
        elif action == 2:
            col = max(0, col - 1)
        elif action == 3:
            col = min(self.cols - 1, col + 1)

        next_state = (row, col)

        # Check obstacles
        if next_state in self.obstacles:
            next_state = self.state  # Stay in place

        self.state = next_state

        # Rewards
        if next_state == self.goal:
            return next_state, 10.0, True
        else:
            return next_state, -0.1, False

    def render(self, q_table=None):
        """Display the grid."""
        action_symbols = ['↑', '↓', '←', '→']
        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                if (r, c) == self.goal:
                    row_str += " G "
                elif (r, c) in self.obstacles:
                    row_str += " # "
                elif q_table and (r, c) in q_table:
                    best = max(range(4), key=lambda a: q_table[(r, c)][a])
                    row_str += f" {action_symbols[best]} "
                else:
                    row_str += " . "
            print(row_str)
        print()


def q_learning(env, episodes=500, alpha=0.1, gamma=0.9,
               epsilon_start=1.0, epsilon_end=0.01):
    """Q-Learning algorithm."""
    q_table = {}
    epsilon = epsilon_start
    epsilon_decay = (epsilon_start - epsilon_end) / episodes

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        for _ in range(100):  # Max steps per episode
            # Initialize Q-values for new state
            if state not in q_table:
                q_table[state] = [0.0] * 4

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(env.get_actions())
            else:
                action = max(range(4), key=lambda a: q_table[state][a])

            # Take action
            next_state, reward, done = env.step(action)
            total_reward += reward

            # Initialize Q-values for new state
            if next_state not in q_table:
                q_table[next_state] = [0.0] * 4

            # Q-Learning update
            best_next = max(q_table[next_state])
            q_table[state][action] += alpha * (
                reward + gamma * best_next - q_table[state][action]
            )

            state = next_state
            if done:
                break

        epsilon -= epsilon_decay

    return q_table


# Driver Code
if __name__ == "__main__":
    print("=== Q-Learning: Grid World ===\n")
    print("Legend: G=Goal, #=Obstacle, Arrows=Best actions\n")

    env = GridWorld(rows=4, cols=4)
    random.seed(42)

    q_table = q_learning(env, episodes=1000, alpha=0.1, gamma=0.9)

    print("Learned Policy:")
    env.render(q_table)

    # Test the learned policy
    print("Test Run (following learned policy):")
    state = env.reset()
    path = [state]
    action_names = ['Up', 'Down', 'Left', 'Right']

    for _ in range(20):
        if state not in q_table:
            break
        action = max(range(4), key=lambda a: q_table[state][a])
        state, reward, done = env.step(action)
        path.append(state)
        if done:
            break

    print(f"  Path: {' -> '.join(str(p) for p in path)}")
    print(f"  Steps: {len(path) - 1}")
    print(f"  Reached goal: {'Yes' if path[-1] == env.goal else 'No'}")
