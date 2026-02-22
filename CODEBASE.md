# Codebase Overview

Technical documentation for developers who want to understand, run, or extend the project.

For the narrative overview of this project, see [README.md](README.md).

---

## Project Structure

```
march-madness-and-chaos/
├── data/
│   └── raw/                              # Kaggle CSVs go here (gitignored)
├── notebooks/
│   ├── act1_autopsy.ipynb                # Act 1: Historical forensic analysis
│   ├── act2_probability.ipynb            # Act 2: Probability model & bridge
│   └── act3_montecarlo.ipynb             # Act 3: Monte Carlo simulation
├── src/
│   ├── __init__.py
│   ├── data_loader.py                    # Data loading, parsing, enrichment
│   ├── upset_analysis.py                 # Act 1 analysis functions
│   ├── probability_model.py              # Act 2 probability model
│   └── simulation.py                     # Act 3 Monte Carlo engine
├── outputs/
│   └── figures/                          # All 18 publication-quality PNGs
├── main.py                               # Smoke test script
├── pyproject.toml                        # Dependencies (uv-managed)
├── CODEBASE.md                           # Technical docs (this file)
└── README.md
```

---

## Module Overview

| Module | Purpose | Key API |
|--------|---------|---------|
| `data_loader.py` | Load & enrich Kaggle CSVs | `MarchMadnessData.get_tourney_results()` |
| `upset_analysis.py` | Historical upset metrics | `upset_rate_by_round()`, `chaos_index()`, `seed_advancement_rates()` |
| `probability_model.py` | Seed-based win probabilities | `build_win_probability_matrix()`, `compute_advancement_probabilities()` |
| `simulation.py` | Monte Carlo bracket simulator | `simulate_tournaments()`, `champion_distribution()`, `final_four_distribution()` |

---

## How It Works

### Data Pipeline

**Raw data** (Kaggle CSVs) &rarr; **MarchMadnessData** (parsing, round inference, enrichment) &rarr; **Analysis modules** (Act 1-3)

The trickiest part of the pipeline is **round inference**. Tournament games don't come with round labels — only `DayNum` (day of the season). We reverse-engineered the round from the fact that the main bracket is always played over exactly 10 game-days in a fixed pattern (32-16-8-4-2-1 games). Days before those 10 are play-in games.

### Probability Model

A deliberately simple **seed-based model** — no team ratings, no ML, no external data:

1. Build a 16x16 **seed-vs-seed win probability matrix** from 39 years of matchup history
2. Use **empirical rates** where we have 10+ historical games for a matchup
3. Fill sparse matchups with a **logistic model** fitted to seed difference (k ~ 0.162)
4. Propagate probabilities through the **fixed bracket structure** (who plays whom depends on who wins earlier rounds)

### Monte Carlo Simulation

Each simulated tournament:
1. Creates **4 independent regions** with the standard seed pairings (1v16, 8v9, 5v12, etc.)
2. Simulates each game by drawing against the win probability matrix
3. Advances winners through R64 &rarr; R32 &rarr; S16 &rarr; E8 per region
4. Pairs region winners for **Final Four** and **Championship**
5. Records: champion seed, Final Four combo, full bracket path

At ~12,000 tournaments/second, 1 million simulations complete in under 2 minutes.

---

## Data Source

This project uses the [Kaggle March Machine Learning Mania](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/data) dataset. The raw CSV files are not included in this repository (they're gitignored) — you'll need to download them from Kaggle and place them in `data/raw/`.

**Minimum required files:**
- `MNCAATourneyCompactResults.csv` — game results (who won, scores)
- `MTeams.csv` — team ID to name mapping
- `MNCAATourneySeeds.csv` — tournament seedings

**Optional files** (used for enrichment but not strictly required):
- `MNCAATourneyDetailedResults.csv` — box score statistics
- `MNCAATourneySlots.csv` — bracket structure
- `MSeasons.csv` — season metadata
- `MTeamConferences.csv` — conference affiliations

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >= 2.2 | Data wrangling |
| numpy | >= 2.0 | Numerical operations |
| matplotlib | >= 3.9 | Static visualizations |
| seaborn | >= 0.13 | Statistical plot styling |
| scipy | >= 1.14 | Logistic model fitting |
| tqdm | >= 4.66 | Progress bars for simulation |
| jupyter | >= 1.0 | Notebook execution |

---

## Quick Start

### Prerequisites

- **Python 3.13+** (managed with [uv](https://docs.astral.sh/uv/))
- **Kaggle dataset**: Download the [March Machine Learning Mania](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/data) CSV files and place them in `data/raw/`

### Setup & Run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/march-madness-and-chaos.git
cd march-madness-and-chaos

# Install dependencies
uv sync

# Quick smoke test — verifies data loads correctly
uv run python main.py

# Open any notebook interactively
uv run jupyter notebook notebooks/act1_autopsy.ipynb
uv run jupyter notebook notebooks/act2_probability.ipynb
uv run jupyter notebook notebooks/act3_montecarlo.ipynb

# Or re-generate all 18 figures headlessly
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/act1_autopsy.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/act2_probability.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/act3_montecarlo.ipynb

# Run the Monte Carlo simulator directly (10K tournament smoke test)
uv run python -m src.simulation
```

> **Note:** Act 3 simulates 1 million tournaments and takes ~2 minutes. The convergence chart runs an additional 1 million simulations internally.

---

## Extending the Project

Some ideas if you want to build on this:

- **Add team-level features**: Incorporate KenPom ratings, conference strength, or preseason rankings for a more granular probability model beyond seed-only
- **Bracket scoring optimization**: Simulate ESPN/Yahoo bracket scoring systems and find the strategy that maximizes expected points
- **Interactive visualizations**: Convert the matplotlib charts to Plotly for hover, zoom, and filter capabilities
- **Yearly predictions**: Plug in the current season's seeds and simulate the upcoming tournament before it starts
- **Parallelize the simulator**: The Monte Carlo engine is pure Python — numpy vectorization or multiprocessing could speed it up 10-100x
- **Historical bracket replay**: Simulate the actual bracket matchups from a specific year and compare simulated outcomes to what really happened
