"""Verify that data loads and core analysis functions work."""

from src.data_loader import MarchMadnessData
from src.upset_analysis import (
    upset_rate_by_round,
    upset_rate_by_matchup,
    chaos_index,
    ROUND_NAMES,
)


def main():
    loader = MarchMadnessData("data/raw")
    games = loader.get_tourney_results()
    print(
        f"Loaded {len(games)} tournament games "
        f"({games['Season'].min()}–{games['Season'].max()})"
    )

    print("\n--- Upset rate by round ---")
    print(upset_rate_by_round(games).to_string(index=False))

    print("\n--- R64 matchup upset rates ---")
    print(upset_rate_by_matchup(games).to_string(index=False))

    print("\n--- Top 10 most chaotic tournaments ---")
    print(chaos_index(games).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
