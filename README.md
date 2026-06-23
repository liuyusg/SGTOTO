# Toto_Singapore

Extraction & Analysis of TOTO Results
A mini-project to analyse 3 years of TOTO Group 1 data.

## Scripts

| Script | Purpose |
|--------|---------|
| `totocreate.py` | Create the SQLite database and tables |
| `totoscrape.py` | Scrape TOTO results from Singapore Pools website |
| `toto_all_locations.py` | Scrape all Singapore Pools branches and authorised sellers |
| `totogeocode.py` | Geocode outlet addresses to latitude/longitude via Google Maps API |
| `totofeatures.py` | Compute and refresh number stats, pair counts, and ML feature tables |
| `totorules.py` | JSON-based rule engine for pattern detection |
| `totobacktest.py` | Strategy backtesting engine |
| `totoml.py` | ML pipeline (baseline, sequence, random-forest models) |

## Setup

**Requirements:** Python 3.9+

```bash
pip install -r requirements.txt
```

Set your Google Maps API key (required for `totogeocode.py`):

```bash
export GOOGLE_API_KEY=your_key_here
```

## Usage

Run the scripts in order:

```bash
python3 totocreate.py --force        # create/reset the database
python3 totoscrape.py                # scrape draw results (caches raw HTML in history/)
python3 toto_all_locations.py        # scrape outlet locations
python3 totogeocode.py               # geocode outlet addresses
python3 totofeatures.py              # refresh feature tables
python3 totorules.py --list          # list configured rules
python3 totobacktest.py --strategy top_frequency  # run a backtest
python3 totoml.py --train --model baseline        # train a model
```

Note: `totocreate.py` drops and recreates all tables.
Omit `--force` to get an interactive confirmation prompt.

### HTML caching

`totoscrape.py` saves each draw page to `history/sppl=<base64>.txt` on first
download. On subsequent runs the cached file is reused, so Singapore Pools is
not re-queried for draws that were already fetched. A cached file that fails the
validity check (missing `drawDate` — e.g. a truncated download) is automatically
deleted and re-fetched.

## Running Tests

Each script has embedded unit tests. Run the full suite (80+ tests):

```bash
python3 -m unittest totoscrape totogeocode toto_all_locations totofeatures totorules totobacktest totoml -v
```

Or run a single script's tests:

```bash
python3 -m unittest totoscrape -v
python3 -m unittest totoml -v
```

## Background

The TOTO $8 Million draw snowballed 3x, coming Monday 18 Oct 9pm. Being a
true-blue Singaporean, I bought some numbers too. We all know this is a game of
chance with low probability of winning, but would it not be interesting to see
some real data?

3 years of Group 1 jackpot data were scraped from Singapore Pools, dumped into a
SQLite database, and geocoded via Google Maps API. The data were then visualised
using Tableau:
https://public.tableau.com/views/Toto102018-21-FINAL/TOTO?:language=en-US&:display_count=n&:origin=viz_share_link
