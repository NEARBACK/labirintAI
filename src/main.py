from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# 1) Среда (лабиринт)
# -------------------------


@dataclass
class MazeEnv:
    grid: np.ndarray  # 0 - пусто, 1 - стена
    start: tuple[int, int]
    goal: tuple[int, int]
    step_penalty: float = -1.0
    wall_penalty: float = -5.0
    goal_reward: float = 50.0
    max_steps: int = 500

    def __post_init__(self):
        self.n_rows, self.n_cols = self.grid.shape
        self.reset()

    def state_to_index(self, r: int, c: int) -> int:
        return r * self.n_cols + c

    def index_to_state(self, idx: int) -> tuple[int, int]:
        return divmod(idx, self.n_cols)

    @property
    def n_states(self) -> int:
        return self.n_rows * self.n_cols

    @property
    def n_actions(self) -> int:
        return 4  # 0: up, 1: right, 2: down, 3: left

    def reset(self) -> int:
        self.pos = tuple(self.start)
        self.steps = 0
        return self.state_to_index(*self.pos)

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        """Совершить действие. Возвращает: next_state, reward, done, info"""
        self.steps += 1
        drc = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dr, dc = drc[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc

        # Проверка границ
        if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols):
            # удар об стену карты -> штраф, позиция не меняется
            reward = self.wall_penalty
            nr, nc = self.pos
        elif self.grid[nr, nc] == 1:
            # внутренняя стена
            reward = self.wall_penalty
            nr, nc = self.pos
        else:
            # успешный шаг
            reward = self.step_penalty

        self.pos = (nr, nc)
        done = False

        if self.pos == self.goal:
            reward += self.goal_reward
            done = True

        if self.steps >= self.max_steps:
            done = True

        s_next = self.state_to_index(*self.pos)
        return s_next, reward, done, {}


# -------------------------
# 2) Агент (Q-Learning)
# -------------------------


@dataclass
class QLearningAgent:
    n_states: int
    n_actions: int
    alpha: float = 0.1  # скорость обучения
    gamma: float = 0.99  # дисконт
    epsilon: float = 1.0  # ε-greedy (исследование)
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995

    def __post_init__(self):
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float32)

    def choose_action(self, state: int) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def learn(self, s: int, a: int, r: float, s_next: int, done: bool):
        best_next = np.max(self.Q[s_next])
        target = r + (0.0 if done else self.gamma * best_next)
        td_error = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# -------------------------
# 3) Обучение и оценка
# -------------------------


def train(env: MazeEnv, agent: QLearningAgent, episodes: int = 1500) -> dict[str, list]:
    rewards = []
    steps_list = []

    for _ in range(episodes):
        s = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            a = agent.choose_action(s)
            s_next, r, done, _ = env.step(a)
            agent.learn(s, a, r, s_next, done)
            s = s_next
            total_reward += r
            steps += 1

        agent.update_epsilon()
        rewards.append(total_reward)
        steps_list.append(steps)

    return {"rewards": rewards, "steps": steps_list}


def greedy_run(env: MazeEnv, Q: np.ndarray, max_steps: int = 500) -> list[tuple[int, int]]:
    """Запуск без исследования: всегда выбираем argmax(Q). Возвращает траекторию."""
    path = [env.start]
    env.reset()
    env.pos = env.start

    for _ in range(max_steps):
        s = env.state_to_index(*env.pos)
        a = int(np.argmax(Q[s]))
        s_next, _, done, _ = env.step(a)
        path.append(env.index_to_state(s_next))
        if done:
            break
    return path


# -------------------------
# 4) Визуализация
# -------------------------


def plot_training_curves(history: dict[str, list], save_path: str | None = None):
    plt.figure()
    plt.plot(history["rewards"])
    plt.title("Суммарная награда по эпизодам")
    plt.xlabel("Эпизод")
    plt.ylabel("Награда")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()


def draw_maze_with_path(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    path: list[tuple[int, int]],
    save_path: str | None = None,
):
    # 0 - пусто (белый), 1 - стена (черный)
    img = np.ones_like(grid, dtype=float)
    img[grid == 1] = 0.0

    plt.figure()
    plt.imshow(img, interpolation="nearest")
    # Отрисуем путь точками
    ys = [p[0] for p in path]
    xs = [p[1] for p in path]
    plt.plot(xs, ys, marker="o", linewidth=1)

    # Старт/Финиш
    plt.scatter([start[1]], [start[0]], marker="s", s=80)
    plt.scatter([goal[1]], [goal[0]], marker="*", s=120)
    plt.title("Путь агента в лабиринте")
    plt.gca().invert_yaxis()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()


# -------------------------
# 5) Пример: запустить из коробки
# -------------------------


def make_default_maze() -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    # Простой лабиринт 10x10
    grid = np.zeros((10, 10), dtype=int)

    # Добавим стенки
    walls = [
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 4),
        (3, 4),
        (4, 4),
        (5, 4),
        (5, 1),
        (5, 2),
        (5, 3),
        (7, 6),
        (7, 7),
        (7, 8),
        (3, 7),
        (4, 7),
        (5, 7),
    ]
    for r, c in walls:
        grid[r, c] = 1

    start = (0, 0)
    goal = (9, 9)
    return grid, start, goal


def main():
    grid, start, goal = make_default_maze()
    env = MazeEnv(grid=grid, start=start, goal=goal, max_steps=300)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=0.1,
        gamma=0.98,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    )

    history = train(env, agent, episodes=1500)

    # Жадный прогон после обучения
    path = greedy_run(env, agent.Q, max_steps=300)

    # Печать краткого отчета
    print(f"Длина пути: {len(path)} шагов")
    print(f"Старт: {start}, Финиш: {goal}")
    print(f"Эпсилон после обучения: {agent.epsilon:.3f}")

    # Визуализация
    plot_training_curves(history, save_path="training_curve.png")
    draw_maze_with_path(grid, start, goal, path, save_path="maze_solution.png")
    print("Сохранено: training_curve.png, maze_solution.png")


if __name__ == "__main__":
    main()
