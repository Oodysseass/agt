import numpy as np


def game_generator(n: int, zero_sum: bool = True) -> tuple[np.ndarray, np.ndarray]:
    A = np.random.rand(n, n)
    if zero_sum:
        return A, -A
    return A, np.random.rand(n, n)


def regret(utilities: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    return max(utilities @ q) - (p @ utilities @ q)


class Player:
    def __init__(self, utilities: np.ndarray):
        self.utilities = utilities
        self.opponent_history = np.zeros(utilities.shape[1])

    def play(self, t: int) -> int:
        if t == 0:
            return np.random.randint(len(self.utilities))
        return int(np.argmax(self.utilities @ (self.opponent_history / t)))

    def update(self, action: int):
        self.opponent_history[action] += 1
