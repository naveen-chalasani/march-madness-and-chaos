"""
Historical upset analysis — upset rates by round, matchup, and season,
chaos rankings, seed advancement rates, and trap seed identification.
"""

import pandas as pd
import numpy as np

from src.data_loader import MarchMadnessData


ROUND_NAMES = {
    0: "Play-in",
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "Championship",
}


def _exclude_playin(games: pd.DataFrame) -> pd.DataFrame:
    """Drop play-in games (Round 0) — they're same-seed matchups."""
    return games[games["Round"] > 0].copy()

# Expected matchups in Round of 64 (seed vs seed)
R64_EXPECTED_MATCHUPS = [
    (1, 16), (2, 15), (3, 14), (4, 13),
    (5, 12), (6, 11), (7, 10), (8, 9),
]


def upset_rate_by_round(games: pd.DataFrame) -> pd.DataFrame:
    """Compute upset rate per round (excluding play-in games).

    An upset is defined as the higher-seeded (numerically larger) team winning.

    Returns DataFrame with: Round, RoundName, TotalGames, Upsets, UpsetRate
    """
    main = _exclude_playin(games)
    by_round = (
        main.groupby("Round")
        .agg(TotalGames=("Upset", "size"), Upsets=("Upset", "sum"))
        .reset_index()
    )
    by_round["UpsetRate"] = by_round["Upsets"] / by_round["TotalGames"]
    by_round["RoundName"] = by_round["Round"].map(ROUND_NAMES)
    return by_round


def upset_rate_by_matchup(games: pd.DataFrame) -> pd.DataFrame:
    """Compute upset rate for each seed-vs-seed matchup in the Round of 64.

    Returns DataFrame with: HighSeed, LowSeed, Matchup (str), Games, Upsets, UpsetRate
    """
    r64 = games[games["Round"] == 1].copy()

    r64["HighSeed"] = r64[["WSeedNum", "LSeedNum"]].min(axis=1)
    r64["LowSeed"] = r64[["WSeedNum", "LSeedNum"]].max(axis=1)

    agg = (
        r64.groupby(["HighSeed", "LowSeed"])
        .agg(Games=("Upset", "size"), Upsets=("Upset", "sum"))
        .reset_index()
    )
    agg["UpsetRate"] = agg["Upsets"] / agg["Games"]
    agg["Matchup"] = agg.apply(
        lambda r: f"#{int(r['HighSeed'])} vs #{int(r['LowSeed'])}", axis=1
    )
    return agg.sort_values("HighSeed").reset_index(drop=True)


def upset_rate_by_year(games: pd.DataFrame) -> pd.DataFrame:
    """Compute upset rate per season with decade grouping (excluding play-ins).

    Returns DataFrame with: Season, TotalGames, Upsets, UpsetRate, Decade
    """
    main = _exclude_playin(games)
    by_year = (
        main.groupby("Season")
        .agg(TotalGames=("Upset", "size"), Upsets=("Upset", "sum"))
        .reset_index()
    )
    by_year["UpsetRate"] = by_year["Upsets"] / by_year["TotalGames"]
    by_year["Decade"] = (by_year["Season"] // 10) * 10
    return by_year


def chaos_index(games: pd.DataFrame) -> pd.DataFrame:
    """Score each tournament by total 'chaos' — a weighted upset metric.

    Each upset is weighted by the magnitude of the seed difference:
        weight = (lower_seed - higher_seed) * round_multiplier

    Round multipliers: later upsets are worth more because they have
    larger bracket-busting impact.

    Returns DataFrame with: Season, ChaosScore, Upsets, TotalGames,
        BiggestUpset (string description)
    """
    round_multiplier = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32}

    main = _exclude_playin(games)
    upsets = main[main["Upset"]].copy()
    upsets["SeedDiff"] = upsets["WSeedNum"] - upsets["LSeedNum"]
    upsets["ChaosWeight"] = (
        upsets["SeedDiff"] * upsets["Round"].map(round_multiplier)
    )

    chaos = (
        upsets.groupby("Season")
        .agg(
            ChaosScore=("ChaosWeight", "sum"),
            Upsets=("Upset", "sum"),
        )
        .reset_index()
    )

    total = main.groupby("Season").size().reset_index(name="TotalGames")
    chaos = chaos.merge(total, on="Season")

    biggest = upsets.loc[upsets.groupby("Season")["ChaosWeight"].idxmax()]
    biggest["BiggestUpset"] = biggest.apply(
        lambda r: (
            f"#{int(r['WSeedNum'])} {r['WTeamName']} over "
            f"#{int(r['LSeedNum'])} {r['LTeamName']} "
            f"({ROUND_NAMES.get(r['Round'], 'R?')})"
        ),
        axis=1,
    )
    chaos = chaos.merge(
        biggest[["Season", "BiggestUpset"]], on="Season", how="left"
    )

    return chaos.sort_values("ChaosScore", ascending=False).reset_index(drop=True)


def seed_advancement_rates(games: pd.DataFrame) -> pd.DataFrame:
    """How often does each seed win in each round?

    Returns a pivot-style DataFrame with SeedNum as index and
    Round (1-6) as columns, values are the fraction of opportunities
    where that seed won (wins / appearances).
    """
    main = _exclude_playin(games)
    n_seasons = main["Season"].nunique()
    n_teams_per_seed = 4  # one per region

    win_rates = main.groupby(["WSeedNum", "Round"]).size().reset_index(name="Wins")
    win_rates["Rate"] = win_rates["Wins"] / (n_seasons * n_teams_per_seed)

    pivot = win_rates.pivot(index="WSeedNum", columns="Round", values="Rate").fillna(0)
    pivot.index.name = "SeedNum"

    return pivot


def trap_seeds(games: pd.DataFrame) -> pd.DataFrame:
    """Identify 'trap picks' — seeds that get upset far more than expected.

    We compare each seed's actual R64 win rate to a naive expectation
    based on seed number. Higher positive 'TrapScore' = worse trap.

    Returns DataFrame with: SeedNum, ExpectedWinRate, ActualWinRate, TrapScore
    """
    r64 = games[games["Round"] == 1].copy()

    r64["FavoredSeed"] = r64[["WSeedNum", "LSeedNum"]].min(axis=1)
    r64["FavoredWon"] = r64["WSeedNum"] < r64["LSeedNum"]

    rates = (
        r64.groupby("FavoredSeed")
        .agg(Games=("FavoredWon", "size"), FavoredWins=("FavoredWon", "sum"))
        .reset_index()
    )
    rates.columns = ["SeedNum", "Games", "FavoredWins"]
    rates["ActualWinRate"] = rates["FavoredWins"] / rates["Games"]

    # Simple expected model: 1 - (seed / 17), so #1 ~ 94%, #8 ~ 53%
    rates["ExpectedWinRate"] = 1 - (rates["SeedNum"] / 17)
    rates["TrapScore"] = rates["ExpectedWinRate"] - rates["ActualWinRate"]

    return rates.sort_values("TrapScore", ascending=False).reset_index(drop=True)
