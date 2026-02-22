"""
Seed-based probability model for March Madness brackets.

Builds a 16x16 win probability matrix from historical data (smoothed
with a logistic model for sparse matchups), computes round-by-round
advancement probabilities, and calculates the bracket possibility space.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.data_loader import MarchMadnessData


def _build_raw_matchup_matrix(games: pd.DataFrame) -> pd.DataFrame:
    """Build a raw win-probability matrix from historical matchup data.

    Returns a 16x16 DataFrame where cell [i, j] = P(seed i beats seed j).
    Cells with no historical data are NaN.
    """
    main = games[games["Round"] > 0].copy()

    main["SeedA"] = main[["WSeedNum", "LSeedNum"]].min(axis=1)
    main["SeedB"] = main[["WSeedNum", "LSeedNum"]].max(axis=1)
    main["SeedAWon"] = main["WSeedNum"] == main["SeedA"]

    agg = (
        main.groupby(["SeedA", "SeedB"])
        .agg(Games=("SeedAWon", "size"), AWins=("SeedAWon", "sum"))
        .reset_index()
    )
    agg["AWinRate"] = agg["AWins"] / agg["Games"]

    matrix = pd.DataFrame(
        np.nan, index=range(1, 17), columns=range(1, 17), dtype=float
    )
    matrix.index.name = "SeedA"
    matrix.columns.name = "SeedB"

    for _, row in agg.iterrows():
        a, b = int(row["SeedA"]), int(row["SeedB"])
        matrix.loc[a, b] = row["AWinRate"]
        matrix.loc[b, a] = 1 - row["AWinRate"]

    for i in range(1, 17):  # same seed vs same seed = coin flip
        matrix.loc[i, i] = 0.5

    return matrix


def _fit_logistic_model(games: pd.DataFrame) -> float:
    """Fit a logistic model: P(lower_seed_wins) = 1 / (1 + exp(-k * seed_diff)).

    Returns the fitted parameter k.
    This is used to fill in sparse cells in the matchup matrix.
    """
    main = games[games["Round"] > 0].copy()
    main["SeedA"] = main[["WSeedNum", "LSeedNum"]].min(axis=1)
    main["SeedB"] = main[["WSeedNum", "LSeedNum"]].max(axis=1)
    main["SeedDiff"] = main["SeedB"] - main["SeedA"]
    main["SeedAWon"] = (main["WSeedNum"] == main["SeedA"]).astype(float)

    def neg_log_likelihood(k: float) -> float:
        p = 1.0 / (1.0 + np.exp(-k * main["SeedDiff"].values))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        y = main["SeedAWon"].values
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    result = minimize_scalar(neg_log_likelihood, bounds=(0.01, 1.0), method="bounded")
    return result.x


def build_win_probability_matrix(
    games: pd.DataFrame,
    min_games: int = 10,
) -> pd.DataFrame:
    """Build a complete 16x16 seed-vs-seed win probability matrix.

    Uses historical data where we have enough samples (>= min_games),
    fills sparse cells with a fitted logistic model based on seed difference.

    Returns a 16x16 DataFrame where cell [i, j] = P(seed i beats seed j).
    Rows and columns are seed numbers 1-16.
    """
    raw = _build_raw_matchup_matrix(games)
    k = _fit_logistic_model(games)

    main = games[games["Round"] > 0]
    game_counts: dict[tuple[int, int], int] = {}
    for i in range(1, 17):
        for j in range(i + 1, 17):
            mask = (
                ((main["WSeedNum"] == i) & (main["LSeedNum"] == j))
                | ((main["WSeedNum"] == j) & (main["LSeedNum"] == i))
            )
            game_counts[(i, j)] = mask.sum()

    matrix = raw.copy()

    for i in range(1, 17):
        for j in range(1, 17):
            if i == j:
                matrix.loc[i, j] = 0.5
                continue

            pair = (min(i, j), max(i, j))
            n_games = game_counts.get(pair, 0)

            if pd.notna(raw.loc[i, j]) and n_games >= min_games:
                continue

            seed_diff = abs(i - j)
            if i < j:
                matrix.loc[i, j] = 1.0 / (1.0 + np.exp(-k * seed_diff))
            else:
                matrix.loc[i, j] = 1.0 / (1.0 + np.exp(k * seed_diff))

    return matrix


BRACKET_R64_MATCHUPS = [
    (1, 16), (8, 9), (5, 12), (4, 13),
    (6, 11), (3, 14), (7, 10), (2, 15),
]


def compute_advancement_probabilities(
    win_prob_matrix: pd.DataFrame,
    n_rounds: int = 6,
) -> pd.DataFrame:
    """Compute probability of each seed reaching (and winning) each round.

    Uses the win probability matrix and the bracket structure to compute
    how likely each seed is to advance through each round.

    Returns a DataFrame with columns: SeedNum, Round, AdvancementProb
    pivoted to SeedNum (index) x Round (columns).
    """
    bracket_slots = [
        (1, 16), (8, 9), (5, 12), (4, 13),
        (6, 11), (3, 14), (7, 10), (2, 15),
    ]

    return _compute_bracket_probs(win_prob_matrix, bracket_slots)


def _merge_slots(
    wp: pd.DataFrame,
    slot_a: dict[int, float],
    slot_b: dict[int, float],
) -> dict[int, float]:
    """Merge two bracket slots: compute P(seed wins) for all seeds in both.

    Given two slots with their seed probability distributions, compute the
    probability that each seed wins the matchup between the two slot winners.

    Both sides can win — seed_a from slot_a beating seed_b from slot_b,
    AND seed_b from slot_b beating seed_a from slot_a.
    """
    result: dict[int, float] = {}
    for seed_a, prob_a in slot_a.items():
        for seed_b, prob_b in slot_b.items():
            # seed_a wins
            p = prob_a * prob_b * wp.loc[seed_a, seed_b]
            result[seed_a] = result.get(seed_a, 0.0) + p
            # seed_b wins
            p = prob_a * prob_b * wp.loc[seed_b, seed_a]
            result[seed_b] = result.get(seed_b, 0.0) + p
    return result


def _compute_bracket_probs(
    wp: pd.DataFrame,
    bracket_slots: list[tuple[int, int]],
) -> pd.DataFrame:
    """Compute advancement probabilities through a quarter-bracket.

    Tracks P(seed s wins round R in this region) through the full bracket.

    Returns a pivot DataFrame: SeedNum (index) x Round (columns) = probability.
    """
    all_seeds = set()
    for a, b in bracket_slots:
        all_seeds.add(a)
        all_seeds.add(b)

    r64_probs = [{} for _ in range(8)]
    for slot_idx, (a, b) in enumerate(bracket_slots):
        r64_probs[slot_idx][a] = wp.loc[a, b]
        r64_probs[slot_idx][b] = wp.loc[b, a]

    r32_probs = [
        _merge_slots(wp, r64_probs[i], r64_probs[i + 1])
        for i in range(0, 8, 2)
    ]

    s16_probs = [
        _merge_slots(wp, r32_probs[i], r32_probs[i + 1])
        for i in range(0, 4, 2)
    ]

    e8_probs = _merge_slots(wp, s16_probs[0], s16_probs[1])

    rounds_data = {
        1: r64_probs,    # 8 slots
        2: r32_probs,    # 4 slots
        3: s16_probs,    # 2 slots
        4: [e8_probs],   # 1 slot
    }

    rows = []
    for seed in sorted(all_seeds):
        for rnd, slot_list in rounds_data.items():
            total_prob = sum(slot.get(seed, 0.0) for slot in slot_list)
            rows.append({"SeedNum": seed, "Round": rnd, "Prob": total_prob})

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="SeedNum", columns="Round", values="Prob").fillna(0)

    # F4 & Championship are cross-region, so we approximate by assuming
    # the opponent is drawn from the E8-winner distribution.
    e8_dist = pivot[4].values
    e8_total = e8_dist.sum()
    e8_dist_norm = e8_dist / e8_total if e8_total > 0 else np.ones_like(e8_dist) / len(e8_dist)

    seeds_sorted = sorted(all_seeds)

    f4_probs = {}
    for seed in seeds_sorted:
        p_reach = pivot.loc[seed, 4]
        p_win_given_reach = sum(
            e8_dist_norm[opp - 1] * wp.loc[seed, opp]
            for opp in seeds_sorted
        )
        f4_probs[seed] = p_reach * p_win_given_reach

    pivot[5] = pd.Series(f4_probs)

    f4_dist = np.array([f4_probs.get(s, 0.0) for s in seeds_sorted])
    f4_total = f4_dist.sum()
    f4_dist_norm = f4_dist / f4_total if f4_total > 0 else np.ones_like(f4_dist) / len(f4_dist)

    champ_probs = {}
    for seed in seeds_sorted:
        p_reach = f4_probs.get(seed, 0.0)
        p_win_given_reach = sum(
            f4_dist_norm[opp - 1] * wp.loc[seed, opp]
            for opp in seeds_sorted
        )
        champ_probs[seed] = p_reach * p_win_given_reach

    pivot[6] = pd.Series(champ_probs)
    pivot = pivot.fillna(0)

    return pivot


def bracket_possibility_space() -> dict[str, int | float]:
    """Calculate the size of the March Madness possibility space.

    Returns a dict with key numbers for the narrative.
    """
    total_brackets = 2**63  # 63 games, each with 2 outcomes

    round_games = {
        "Round of 64": 32,
        "Round of 32": 16,
        "Sweet 16": 8,
        "Elite 8": 4,
        "Final Four": 2,
        "Championship": 1,
    }
    round_combos = {name: 2**n for name, n in round_games.items()}

    cumulative = {}
    running = 1
    for name, n in round_games.items():
        running *= 2**n
        cumulative[name] = running

    return {
        "total_brackets": total_brackets,
        "total_brackets_scientific": f"{total_brackets:.2e}",
        "total_games": 63,
        "round_games": round_games,
        "round_combinations": round_combos,
        "cumulative_after_round": cumulative,
    }


def format_big_number(n: int | float) -> str:
    """Format a large number for narrative display."""
    if n >= 1e18:
        return f"{n / 1e18:.1f} quintillion"
    elif n >= 1e15:
        return f"{n / 1e15:.1f} quadrillion"
    elif n >= 1e12:
        return f"{n / 1e12:.1f} trillion"
    elif n >= 1e9:
        return f"{n / 1e9:.1f} billion"
    elif n >= 1e6:
        return f"{n / 1e6:.1f} million"
    else:
        return f"{n:,.0f}"


if __name__ == "__main__":
    loader = MarchMadnessData("data/raw")
    games = loader.get_tourney_results()

    print("=== Fitted logistic parameter ===")
    k = _fit_logistic_model(games)
    print(f"k = {k:.4f}")
    print(f"Example: P(#1 beats #16) = {1/(1+np.exp(-k*15)):.4f}")
    print(f"Example: P(#5 beats #12) = {1/(1+np.exp(-k*7)):.4f}")
    print(f"Example: P(#8 beats #9)  = {1/(1+np.exp(-k*1)):.4f}")

    print("\n=== Win probability matrix (selected) ===")
    wp = build_win_probability_matrix(games)
    seeds_to_show = [1, 2, 3, 4, 5, 8, 11, 16]
    print(wp.loc[seeds_to_show, seeds_to_show].round(3).to_string())

    print("\n=== Advancement probabilities (per region) ===")
    adv = compute_advancement_probabilities(wp)
    print(adv.round(4).to_string())

    print("\n=== Possibility space ===")
    space = bracket_possibility_space()
    print(f"Total possible brackets: {format_big_number(space['total_brackets'])}")
    for name, combos in space["cumulative_after_round"].items():
        print(f"  After {name}: {format_big_number(combos)}")
