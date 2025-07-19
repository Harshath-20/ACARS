import random
import numpy as np

class QLearningAgent:
    def __init__(self, num_items, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.q_table = {}  # Dictionary: {(user_state, item_id): Q-value}
        self.num_items = num_items
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor

    def get_state_action_key(self, user_state, item_id):
        return (tuple(user_state.tolist()), item_id)

    def select_action(self, user_state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_items - 1)  # Explore
        else:
            # Exploit: choose item with highest Q-value
            q_values = [self.q_table.get(self.get_state_action_key(user_state, i), 0.0)
                        for i in range(self.num_items)]
            return int(np.argmax(q_values))

    def update(self, user_state, item_id, reward, next_state):
        key = self.get_state_action_key(user_state, item_id)
        max_future_q = max([self.q_table.get(self.get_state_action_key(next_state, a), 0.0)
                            for a in range(self.num_items)])
        current_q = self.q_table.get(key, 0.0)
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[key] = new_q
