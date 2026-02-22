"""
Monte Carlo bracket simulator.

Uses the seed-based win probability matrix to simulate full 64-team
tournaments at scale — champion distribution, Final Four combinations,
upset frequencies, and convergence behavior.
"""

import numpy as np
import pandas as pd
from collections import Counter

from src.probability_model import BRACKET_R64_MATCHUPS


def _simulate_game(
    seed_a: int,
    seed_b: int,
    wp_lookup: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Simulate a single game. Returns the winning seed."""
    p_a_wins = wp_lookup[seed_a - 1, seed_b - 1]
    return seed_a if rng.random() < p_a_wins else seed_b


def _simulate_region(
    wp_lookup: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Simulate one region (R64 through E8). Returns round-by-round results.

    Returns a dict with:
      - 'r64_winners': list of 8 seeds (winners of R64)
      - 'r32_winners': list of 4 seeds (winners of R32)
      - 's16_winners': list of 2 seeds (winners of S16)
      - 'e8_winner':   single seed (region champion)
    """
    r64_winners = []
    for seed_a, seed_b in BRACKET_R64_MATCHUPS:
        r64_winners.append(_simulate_game(seed_a, seed_b, wp_lookup, rng))

    r32_winners = []
    for i in range(0, 8, 2):
        r32_winners.append(
            _simulate_game(r64_winners[i], r64_winners[i + 1], wp_lookup, rng)
        )

    s16_winners = []
    for i in range(0, 4, 2):
        s16_winners.append(
            _simulate_game(r32_winners[i], r32_winners[i + 1], wp_lookup, rng)
        )

    e8_winner = _simulate_game(s16_winners[0], s16_winners[1], wp_lookup, rng)

    return {
        "r64_winners": r64_winners,
        "r32_winners": r32_winners,
        "s16_winners": s16_winners,
        "e8_winner": e8_winner,
    }


def _simulate_one_tournament(
    wp_lookup: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Simulate a full 64-team tournament. Returns structured results.

    Returns a dict with:
      - 'regions': list of 4 region results
      - 'final_four': tuple of 4 seeds (region winners)
      - 'f4_winners': tuple of 2 seeds (F4 game winners)
      - 'champion': int (winning seed)
      - 'runner_up': int (losing finalist seed)
    """
    regions = [_simulate_region(wp_lookup, rng) for _ in range(4)]

    final_four = tuple(r["e8_winner"] for r in regions)

    f4_winner_1 = _simulate_game(final_four[0], final_four[1], wp_lookup, rng)
    f4_winner_2 = _simulate_game(final_four[2], final_four[3], wp_lookup, rng)

    champion = _simulate_game(f4_winner_1, f4_winner_2, wp_lookup, rng)
    runner_up = f4_winner_2 if champion == f4_winner_1 else f4_winner_1

    return {
        "regions": regions,
        "final_four": final_four,
        "f4_winners": (f4_winner_1, f4_winner_2),
        "champion": champion,
        "runner_up": runner_up,
    }


def simulate_tournaments(
    win_prob_matrix: pd.DataFrame,
    n: int = 100_000,
    seed: int | None = 42,
    show_progress: bool = True,
) -> list[dict]:
    """Simulate n full tournaments and return results.

    Parameters
    ----------
    win_prob_matrix : pd.DataFrame
        16x16 win probability matrix from probability_model.build_win_probability_matrix().
    n : int
        Number of tournaments to simulate.
    seed : int or None
        Random seed for reproducibility. None = random.
    show_progress : bool
        Show tqdm progress bar.

    Returns
    -------
    list[dict]
        List of tournament result dicts (one per simulation).
    """
    rng = np.random.default_rng(seed)

    wp_lookup = win_prob_matrix.values.copy()

    results = []

    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(range(n), desc="Simulating tournaments", unit="tourney")
        except ImportError:
            iterator = range(n)
    else:
        iterator = range(n)

    for _ in iterator:
        results.append(_simulate_one_tournament(wp_lookup, rng))

    return results


def champion_distribution(results: list[dict]) -> pd.DataFrame:
    """Count how often each seed won the championship.

    Returns DataFrame with columns: Seed, Count, Percentage
    sorted by Seed.
    """
    champs = [r["champion"] for r in results]
    counter = Counter(champs)
    n = len(results)

    rows = []
    for seed in range(1, 17):
        count = counter.get(seed, 0)
        rows.append({
            "Seed": seed,
            "Count": count,
            "Percentage": count / n if n > 0 else 0,
        })

    return pd.DataFrame(rows)


def final_four_distribution(results: list[dict]) -> pd.DataFrame:
    """Analyze Final Four seed combinations.

    Returns DataFrame with columns: Combo (sorted tuple), Count, Percentage
    sorted by Count descending.
    """
    combos = [tuple(sorted(r["final_four"])) for r in results]
    counter = Counter(combos)
    n = len(results)

    rows = [
        {"Combo": combo, "Count": count, "Percentage": count / n}
        for combo, count in counter.most_common()
    ]

    return pd.DataFrame(rows)


def round_survival_rates(results: list[dict]) -> pd.DataFrame:
    """Compute simulated survival rates for each seed through each round.

    Returns a DataFrame pivoted: SeedNum (index) x Round (columns) = rate.
    """
    seed_round_counts = {}
    n = len(results)

    for r in results:
        for region in r["regions"]:
            for s in region["r64_winners"]:
                seed_round_counts[(s, 1)] = seed_round_counts.get((s, 1), 0) + 1
            for s in region["r32_winners"]:
                seed_round_counts[(s, 2)] = seed_round_counts.get((s, 2), 0) + 1
            for s in region["s16_winners"]:
                seed_round_counts[(s, 3)] = seed_round_counts.get((s, 3), 0) + 1
            seed_round_counts[(region["e8_winner"], 4)] = \
                seed_round_counts.get((region["e8_winner"], 4), 0) + 1

        for s in r["f4_winners"]:
            seed_round_counts[(s, 5)] = seed_round_counts.get((s, 5), 0) + 1

        seed_round_counts[(r["champion"], 6)] = \
            seed_round_counts.get((r["champion"], 6), 0) + 1

    denominator = n * 4  # 4 regions per tournament

    rows = []
    for seed in range(1, 17):
        for rnd in range(1, 7):
            count = seed_round_counts.get((seed, rnd), 0)
            rows.append({
                "SeedNum": seed,
                "Round": rnd,
                "Rate": count / denominator if denominator > 0 else 0,
            })

    df = pd.DataFrame(rows)
    return df.pivot(index="SeedNum", columns="Round", values="Rate").fillna(0)


def upset_frequency(results: list[dict]) -> pd.DataFrame:
    """Track upset frequency per round across simulations.

    Returns DataFrame with columns: Round, TotalGames, Upsets, UpsetRate
    """
    round_games = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    round_upsets = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

    for r in results:
        for region in r["regions"]:
            for idx, (seed_a, seed_b) in enumerate(BRACKET_R64_MATCHUPS):
                winner = region["r64_winners"][idx]
                round_games[1] += 1
                if winner == max(seed_a, seed_b):
                    round_upsets[1] += 1

            for i in range(0, 8, 2):
                sa, sb = region["r64_winners"][i], region["r64_winners"][i + 1]
                winner = region["r32_winners"][i // 2]
                round_games[2] += 1
                if sa != sb and winner == max(sa, sb):
                    round_upsets[2] += 1

            for i in range(0, 4, 2):
                sa, sb = region["r32_winners"][i], region["r32_winners"][i + 1]
                winner = region["s16_winners"][i // 2]
                round_games[3] += 1
                if sa != sb and winner == max(sa, sb):
                    round_upsets[3] += 1

            sa, sb = region["s16_winners"][0], region["s16_winners"][1]
            winner = region["e8_winner"]
            round_games[4] += 1
            if sa != sb and winner == max(sa, sb):
                round_upsets[4] += 1

        ff = r["final_four"]
        sa, sb = ff[0], ff[1]
        winner = r["f4_winners"][0]
        round_games[5] += 1
        if sa != sb and winner == max(sa, sb):
            round_upsets[5] += 1

        sa, sb = ff[2], ff[3]
        winner = r["f4_winners"][1]
        round_games[5] += 1
        if sa != sb and winner == max(sa, sb):
            round_upsets[5] += 1

        sa, sb = r["f4_winners"][0], r["f4_winners"][1]
        winner = r["champion"]
        round_games[6] += 1
        if sa != sb and winner == max(sa, sb):
            round_upsets[6] += 1

    rows = [
        {
            "Round": rnd,
            "TotalGames": round_games[rnd],
            "Upsets": round_upsets[rnd],
            "UpsetRate": round_upsets[rnd] / round_games[rnd] if round_games[rnd] > 0 else 0,
        }
        for rnd in range(1, 7)
    ]
    return pd.DataFrame(rows)


def cinderella_runs(results: list[dict], min_seed: int = 7) -> pd.DataFrame:
    """Track how far "Cinderella" seeds (>= min_seed) advance.

    Returns DataFrame with columns: Seed, FurthestRound, Count, Percentage
    """
    furthest = []

    for r in results:
        for region in r["regions"]:
            for s in set(region["r64_winners"]):
                if s >= min_seed:
                    max_rnd = 1
                    if s in region["r32_winners"]:
                        max_rnd = 2
                    if s in region["s16_winners"]:
                        max_rnd = 3
                    if s == region["e8_winner"]:
                        max_rnd = 4
                    furthest.append({"Seed": s, "FurthestRound": max_rnd})

        for s in r["f4_winners"]:
            if s >= min_seed:
                furthest.append({"Seed": s, "FurthestRound": 5})

        if r["champion"] >= min_seed:
            furthest.append({"Seed": r["champion"], "FurthestRound": 6})

    if not furthest:
        return pd.DataFrame(columns=["Seed", "FurthestRound", "Count", "Percentage"])

    df = pd.DataFrame(furthest)
    counts = df.groupby(["Seed", "FurthestRound"]).size().reset_index(name="Count")
    n = len(results) * 4  # 4 regions
    counts["Percentage"] = counts["Count"] / n

    return counts.sort_values(["Seed", "FurthestRound"]).reset_index(drop=True)


def convergence_data(
    win_prob_matrix: pd.DataFrame,
    checkpoints: list[int] | None = None,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Run simulations at increasing sizes and track champion distribution convergence.

    Useful for showing how the distribution stabilizes as N grows.

    Parameters
    ----------
    win_prob_matrix : pd.DataFrame
        16x16 win probability matrix.
    checkpoints : list[int]
        Number of simulations at each checkpoint.
    seed : int or None
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: N, Seed, Percentage (champion percentage at that checkpoint)
    """
    if checkpoints is None:
        checkpoints = [100, 500, 1000, 5000, 10_000, 50_000, 100_000, 500_000, 1_000_000]

    rng = np.random.default_rng(seed)
    wp_lookup = win_prob_matrix.values.copy()

    max_n = max(checkpoints)
    champions = []

    try:
        from tqdm import tqdm
        iterator = tqdm(range(max_n), desc="Convergence simulation", unit="tourney")
    except ImportError:
        iterator = range(max_n)

    for _ in iterator:
        result = _simulate_one_tournament(wp_lookup, rng)
        champions.append(result["champion"])

    rows = []
    for cp in checkpoints:
        subset = champions[:cp]
        counter = Counter(subset)
        for s in range(1, 17):
            rows.append({
                "N": cp,
                "Seed": s,
                "Percentage": counter.get(s, 0) / cp,
            })

    return pd.DataFrame(rows)


def match_historical_champions(
    results: list[dict],
    historical_games: pd.DataFrame,
) -> pd.DataFrame:
    """Check how often simulations produce the same champion seed as real history.

    Parameters
    ----------
    results : list[dict]
        Simulation results.
    historical_games : pd.DataFrame
        From data_loader.get_tourney_results().

    Returns
    -------
    pd.DataFrame
        Columns: Season, HistoricalChampSeed, SimMatchRate, FirstMatch
    """
    champ_games = historical_games[historical_games["Round"] == 6].copy()
    hist_champs = champ_games.set_index("Season")["WSeedNum"].to_dict()

    sim_champs = [r["champion"] for r in results]
    n = len(sim_champs)

    rows = []
    for season, hist_seed in sorted(hist_champs.items()):
        matches = [i for i, s in enumerate(sim_champs) if s == hist_seed]
        match_rate = len(matches) / n if n > 0 else 0
        first_match = matches[0] + 1 if matches else None

        rows.append({
            "Season": season,
            "HistoricalChampSeed": hist_seed,
            "SimMatchRate": match_rate,
            "FirstMatch": first_match,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.data_loader import MarchMadnessData
    from src.probability_model import build_win_probability_matrix

    loader = MarchMadnessData("data/raw")
    games = loader.get_tourney_results()
    wp = build_win_probability_matrix(games)

    print("=== Running 10,000 tournament simulations ===")
    results = simulate_tournaments(wp, n=10_000, seed=42)

    print("\n=== Champion seed distribution ===")
    champ_dist = champion_distribution(results)
    for _, row in champ_dist.iterrows():
        bar = "#" * int(row["Percentage"] * 200)
        print(f"  #{int(row['Seed']):>2}: {row['Percentage']:6.2%}  {bar}")

    print("\n=== Top 10 Final Four combinations ===")
    f4_dist = final_four_distribution(results)
    for _, row in f4_dist.head(10).iterrows():
        seeds = ", ".join(f"#{s}" for s in row["Combo"])
        print(f"  ({seeds}): {row['Count']:,} times ({row['Percentage']:.2%})")

    print("\n=== Simulated upset rates by round ===")
    upsets = upset_frequency(results)
    from src.upset_analysis import ROUND_NAMES
    for _, row in upsets.iterrows():
        rnd = int(row["Round"])
        print(f"  {ROUND_NAMES.get(rnd, f'R{rnd}')}: {row['UpsetRate']:.1%}")
