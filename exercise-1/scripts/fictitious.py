import argparse
import itertools
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from helpers import game_generator, regret, Player

T_CAP = 1_000_000


def play_game(A: np.ndarray, B: np.ndarray, T: int, epsilon: float) -> tuple[bool, int]:
    player1 = Player(A)
    player2 = Player(B.T)

    for t in range(1, T + 1):
        action1 = player1.play(t - 1)
        action2 = player2.play(t - 1)
        player1.update(action2)
        player2.update(action1)

        emp1 = player2.opponent_history / t
        emp2 = player1.opponent_history / t

        if regret(A, emp1, emp2) <= epsilon and regret(B.T, emp2, emp1) <= epsilon:
            return True, t

    return False, T


def load_or_generate(path: Path, n: int, zero_sum: bool) -> tuple[np.ndarray, np.ndarray]:
    if path.exists():
        data = np.load(path)
        return data['A'], data['B']
    A, B = game_generator(n, zero_sum)
    np.savez(path, A=A, B=B)
    return A, B


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, nargs='+', default=[4])
    parser.add_argument('--epsilon', type=float, nargs='+', default=[0.01])
    parser.add_argument('--games', type=int, default=10)
    parser.add_argument('--zero_sum', action='store_true')
    args = parser.parse_args()

    base = Path(__file__).parent
    games_dir = base / 'games'
    results_dir = base / 'results'
    games_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report = {
        "timestamp": timestamp,
        "config": {
            "n_values": args.n,
            "epsilon_values": args.epsilon,
            "games_per_combination": args.games,
            "zero_sum": args.zero_sum,
            "T_cap": T_CAP,
        },
        "results": []
    }

    for n, epsilon in itertools.product(args.n, args.epsilon):
        T_max = min(2 ** n, T_CAP)
        game_results = []

        for i in range(args.games):
            path = games_dir / f'n{n}_zerosum{args.zero_sum}_{i}.npz'
            A, B = load_or_generate(path, n, args.zero_sum)
            converged, iterations = play_game(A, B, T_max, epsilon)
            game_results.append({"game": i, "converged": converged, "iterations": iterations})

        n_converged = sum(r["converged"] for r in game_results)
        all_iters = [r["iterations"] for r in game_results]
        conv_iters = [r["iterations"] for r in game_results if r["converged"]]

        summary = {
            "n": n,
            "epsilon": epsilon,
            "T_max": T_max,
            "converged": f"{n_converged}/{args.games}",
            "convergence_rate": round(n_converged / args.games, 2),
            "avg_iterations": round(float(np.mean(all_iters)), 1),
            "avg_iterations_when_converged": round(float(np.mean(conv_iters)), 1) if conv_iters else None,
            "games": game_results,
        }
        report["results"].append(summary)

        print(f"n={n:>4}  epsilon={epsilon}  T_max={T_max:>9}  zero_sum={args.zero_sum}")
        print(f"  converged: {n_converged}/{args.games}  "
              f"avg iters: {np.mean(all_iters):>10.1f}  "
              f"avg iters (converged): {np.mean(conv_iters) if conv_iters else float('nan'):>10.1f}")

    out_path = results_dir / f'run_{timestamp}.json'
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
