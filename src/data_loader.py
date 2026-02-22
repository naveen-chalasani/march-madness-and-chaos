"""
Data loader for Kaggle March Machine Learning Mania dataset.

Expects CSV files in data/raw/ with the M-prefix naming convention
(e.g., MNCAATourneyCompactResults.csv, MTeams.csv, MNCAATourneySeeds.csv).

Usage:
    loader = MarchMadnessData("data/raw")
    games = loader.get_tourney_results()      # enriched with seeds & team names
    seeds = loader.get_seeds()                # parsed seed info
"""

from pathlib import Path

import pandas as pd


class MarchMadnessData:
    """Loads, parses, and joins the core March Madness CSVs."""

    # Files we care about for this project
    _FILES = {
        "teams": "MTeams.csv",
        "seasons": "MSeasons.csv",
        "tourney_compact": "MNCAATourneyCompactResults.csv",
        "tourney_detailed": "MNCAATourneyDetailedResults.csv",
        "seeds": "MNCAATourneySeeds.csv",
        "slots": "MNCAATourneySlots.csv",
        "conferences": "MTeamConferences.csv",
    }

    # 64-team bracket era begins in 1985
    BRACKET_ERA_START = 1985

    # Expected game counts per round in the main bracket
    _ROUND_GAME_COUNTS = {
        1: 32,  # Round of 64
        2: 16,  # Round of 32
        3: 8,   # Sweet 16
        4: 4,   # Elite 8
        5: 2,   # Final Four
        6: 1,   # Championship
    }

    def __init__(self, data_dir: str | Path = "data/raw"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir.resolve()}\n"
                "Download the Kaggle March Machine Learning Mania dataset "
                "and place the CSV files in this directory."
            )
        self._cache: dict[str, pd.DataFrame] = {}

    def _load(self, key: str) -> pd.DataFrame:
        if key not in self._cache:
            path = self.data_dir / self._FILES[key]
            if not path.exists():
                raise FileNotFoundError(f"Missing required file: {path}")
            self._cache[key] = pd.read_csv(path)
        return self._cache[key].copy()

    def load_teams(self) -> pd.DataFrame:
        return self._load("teams")

    def load_seasons(self) -> pd.DataFrame:
        return self._load("seasons")

    def load_tourney_compact(self) -> pd.DataFrame:
        return self._load("tourney_compact")

    def load_tourney_detailed(self) -> pd.DataFrame:
        return self._load("tourney_detailed")

    def load_seeds_raw(self) -> pd.DataFrame:
        return self._load("seeds")

    def load_slots(self) -> pd.DataFrame:
        return self._load("slots")

    def load_conferences(self) -> pd.DataFrame:
        return self._load("conferences")

    def get_seeds(self) -> pd.DataFrame:
        """Parse seed strings into region, seed number, and play-in flag.

        Returns DataFrame with columns:
            Season, TeamID, Seed (original string),
            Region (W/X/Y/Z), SeedNum (1-16), PlayIn (bool)
        """
        seeds = self.load_seeds_raw()
        seeds["Region"] = seeds["Seed"].str[0]
        seeds["SeedNum"] = seeds["Seed"].str[1:3].astype(int)
        seeds["PlayIn"] = seeds["Seed"].str.len() > 3
        return seeds

    def _team_name_map(self) -> dict[int, str]:
        teams = self.load_teams()
        return dict(zip(teams["TeamID"], teams["TeamName"]))

    def _infer_round(self, df: pd.DataFrame) -> pd.Series:
        """Infer tournament round from DayNum within each season.

        Strategy: the main bracket is played over exactly 10 days in a
        fixed pattern (working backwards from the championship):

            Championship:   1 game  on 1 day   (last day)
            Final Four:     2 games on 1 day   (2nd-to-last day)
            Elite 8:        4 games on 2 days
            Sweet 16:       8 games on 2 days
            Round of 32:   16 games on 2 days
            Round of 64:   32 games on 2 days

        Any days before those 10 are play-in games (Round 0).

        Returns a Series with values 0-6:
            0 = Play-in, 1 = Round of 64, 2 = Round of 32,
            3 = Sweet 16, 4 = Elite 8, 5 = Final Four,
            6 = Championship
        """
        round_series = pd.Series(0, index=df.index, dtype=int)

        _OFFSET_TO_ROUND = {
            1: 6,   # last day
            2: 5,   # 2nd from end
            3: 4, 4: 4,
            5: 3, 6: 3,
            7: 2, 8: 2,
            9: 1, 10: 1,
        }

        for _, group in df.groupby("Season"):
            sorted_days = sorted(group["DayNum"].unique(), reverse=True)

            day_to_round: dict[int, int] = {}
            for i, day in enumerate(sorted_days):
                offset = i + 1
                day_to_round[day] = _OFFSET_TO_ROUND.get(offset, 0)

            for idx in group.index:
                round_series.loc[idx] = day_to_round[df.loc[idx, "DayNum"]]

        return round_series

    def get_tourney_results(self, detailed: bool = False) -> pd.DataFrame:
        """Load tournament results enriched with seeds, team names, and round.

        Returns a DataFrame with one row per game, columns including:
            Season, Round, DayNum,
            WTeamID, WTeamName, WSeed, WSeedNum,
            LTeamID, LTeamName, LSeed, LSeedNum,
            WScore, LScore, ScoreDiff, NumOT,
            Upset (bool — True when lower seed beats higher seed)
        """
        if detailed:
            games = self.load_tourney_detailed()
        else:
            games = self.load_tourney_compact()

        games = games[games["Season"] >= self.BRACKET_ERA_START].copy()
        games["Round"] = self._infer_round(games)

        seeds = self.get_seeds()
        seed_cols = ["Season", "TeamID", "Seed", "SeedNum", "Region"]

        games = games.merge(
            seeds[seed_cols].rename(
                columns={
                    "TeamID": "WTeamID",
                    "Seed": "WSeed",
                    "SeedNum": "WSeedNum",
                    "Region": "WRegion",
                }
            ),
            on=["Season", "WTeamID"],
            how="left",
        )
        games = games.merge(
            seeds[seed_cols].rename(
                columns={
                    "TeamID": "LTeamID",
                    "Seed": "LSeed",
                    "SeedNum": "LSeedNum",
                    "Region": "LRegion",
                }
            ),
            on=["Season", "LTeamID"],
            how="left",
        )

        name_map = self._team_name_map()
        games["WTeamName"] = games["WTeamID"].map(name_map)
        games["LTeamName"] = games["LTeamID"].map(name_map)

        games["ScoreDiff"] = games["WScore"] - games["LScore"]
        games["Upset"] = games["WSeedNum"] > games["LSeedNum"]

        games = games.sort_values(["Season", "DayNum"]).reset_index(drop=True)

        return games

    def get_matchup_history(self) -> pd.DataFrame:
        """Reshape results so each row represents one team's perspective.

        Returns a DataFrame with columns:
            Season, Round, TeamID, TeamName, Seed, SeedNum,
            OpponentID, OpponentName, OpponentSeed, OpponentSeedNum,
            Won (bool), Score, OpponentScore
        """
        games = self.get_tourney_results()
        rows = []

        for _, g in games.iterrows():
            base = {"Season": g["Season"], "Round": g["Round"]}
            # Winner row
            rows.append(
                {
                    **base,
                    "TeamID": g["WTeamID"],
                    "TeamName": g["WTeamName"],
                    "Seed": g["WSeed"],
                    "SeedNum": g["WSeedNum"],
                    "OpponentID": g["LTeamID"],
                    "OpponentName": g["LTeamName"],
                    "OpponentSeed": g["LSeed"],
                    "OpponentSeedNum": g["LSeedNum"],
                    "Won": True,
                    "Score": g["WScore"],
                    "OpponentScore": g["LScore"],
                }
            )
            # Loser row
            rows.append(
                {
                    **base,
                    "TeamID": g["LTeamID"],
                    "TeamName": g["LTeamName"],
                    "Seed": g["LSeed"],
                    "SeedNum": g["LSeedNum"],
                    "OpponentID": g["WTeamID"],
                    "OpponentName": g["WTeamName"],
                    "OpponentSeed": g["WSeed"],
                    "OpponentSeedNum": g["WSeedNum"],
                    "Won": False,
                    "Score": g["LScore"],
                    "OpponentScore": g["WScore"],
                }
            )

        return pd.DataFrame(rows)

    def get_seed_win_rates(self) -> pd.DataFrame:
        """Compute historical win rate for each seed in each round.

        Returns a DataFrame with columns:
            SeedNum, Round, Wins, Losses, Games, WinRate
        """
        matchups = self.get_matchup_history()

        agg = (
            matchups.groupby(["SeedNum", "Round"])["Won"]
            .agg(["sum", "count"])
            .reset_index()
        )
        agg.columns = ["SeedNum", "Round", "Wins", "Games"]
        agg["Losses"] = agg["Games"] - agg["Wins"]
        agg["WinRate"] = agg["Wins"] / agg["Games"]

        return agg.sort_values(["Round", "SeedNum"]).reset_index(drop=True)


if __name__ == "__main__":
    loader = MarchMadnessData()
    print("--- Teams ---")
    print(loader.load_teams().head())
    print(f"\n--- Seeds ({loader.get_seeds().shape[0]} rows) ---")
    print(loader.get_seeds().head(10))
    print(f"\n--- Tournament games ({loader.get_tourney_results().shape[0]} games) ---")
    print(loader.get_tourney_results().head(10))
    print("\n--- Seed win rates ---")
    print(loader.get_seed_win_rates().head(20))
