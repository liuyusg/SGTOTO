import sqlite3
import sys

######### CREATE SQLITE TABLE ############

# Guard against accidental data loss
if '--force' not in sys.argv:
    confirm = input('This will DROP all existing tables and data. Type YES to continue: ')
    if confirm.strip() != 'YES':
        print('Aborted.')
        sys.exit(0)

conn = sqlite3.connect('toto.sqlite')
try:
    cur = conn.cursor()
    cur.executescript('''
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS ml_features;
DROP TABLE IF EXISTS backtest_results;
DROP TABLE IF EXISTS backtests;
DROP TABLE IF EXISTS rule_results;
DROP TABLE IF EXISTS rules;
DROP TABLE IF EXISTS number_pairs;
DROP TABLE IF EXISTS number_stats;
DROP TABLE IF EXISTS draw_prizes;
DROP TABLE IF EXISTS winners;
DROP TABLE IF EXISTS place;
DROP TABLE IF EXISTS jackpot_no;
DROP TABLE IF EXISTS draws;
DROP TABLE IF EXISTS placeall;

CREATE TABLE draws (
    draw_no         INTEGER PRIMARY KEY,
    day             TEXT,
    date            TEXT UNIQUE,
    draw_type       TEXT DEFAULT 'ordinary',
    scanned         INTEGER DEFAULT 0
);

CREATE TABLE jackpot_no (
    draw_no         INTEGER,
    no_type         TEXT,
    number          INTEGER,
    UNIQUE(draw_no, no_type, number),
    FOREIGN KEY(draw_no) REFERENCES draws(draw_no)
);

CREATE TABLE draw_prizes (
    draw_no         INTEGER,
    prize_group     INTEGER,
    prize_amount    INTEGER,
    winner_count    INTEGER,
    UNIQUE(draw_no, prize_group),
    FOREIGN KEY(draw_no) REFERENCES draws(draw_no)
);

CREATE TABLE place (
    draw_no         INTEGER,
    raw_data        TEXT,
    location        TEXT,
    address         TEXT,
    quickpick       TEXT,
    system          TEXT,
    UNIQUE(draw_no, location, address),
    FOREIGN KEY(draw_no) REFERENCES draws(draw_no)
);

CREATE TABLE winners (
    draw_no         INTEGER,
    prize_group     INTEGER,
    raw_data        TEXT,
    location        TEXT,
    address         TEXT,
    quickpick       TEXT,
    system          TEXT,
    UNIQUE(draw_no, prize_group, location, address),
    FOREIGN KEY(draw_no) REFERENCES draws(draw_no)
);

CREATE TABLE placeall (
    location        TEXT,
    address         TEXT,
    postal_code     TEXT,
    latitude        FLOAT,
    longitude       FLOAT,
    scanned         INTEGER DEFAULT 0
);

-- Pre-computed feature store (refreshed by totofeatures.py)
CREATE TABLE number_stats (
    number          INTEGER PRIMARY KEY,
    appearances     INTEGER DEFAULT 0,
    last_draw_no    INTEGER,
    avg_gap         REAL,
    max_gap         INTEGER
);

CREATE TABLE number_pairs (
    number_a        INTEGER,
    number_b        INTEGER,
    co_appearances  INTEGER DEFAULT 0,
    PRIMARY KEY(number_a, number_b),
    CHECK(number_a < number_b)
);

-- Rule engine
CREATE TABLE rules (
    rule_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    description     TEXT,
    rule_json       TEXT NOT NULL,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE rule_results (
    rule_id         INTEGER,
    draw_no         INTEGER,
    matched         INTEGER NOT NULL,
    details_json    TEXT,
    PRIMARY KEY(rule_id, draw_no),
    FOREIGN KEY(rule_id) REFERENCES rules(rule_id),
    FOREIGN KEY(draw_no) REFERENCES draws(draw_no)
);

-- Backtesting engine
CREATE TABLE backtests (
    backtest_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    strategy_json   TEXT NOT NULL,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE backtest_results (
    backtest_id         INTEGER,
    draw_no             INTEGER,
    picked_numbers_json TEXT,
    matched_count       INTEGER,
    prize_group         INTEGER,
    prize_amount        INTEGER,
    PRIMARY KEY(backtest_id, draw_no),
    FOREIGN KEY(backtest_id) REFERENCES backtests(backtest_id),
    FOREIGN KEY(draw_no)     REFERENCES draws(draw_no)
);

-- ML pipeline
CREATE TABLE ml_features (
    draw_no         INTEGER PRIMARY KEY,
    features_json   TEXT NOT NULL,
    FOREIGN KEY(draw_no) REFERENCES draws(draw_no)
);

CREATE TABLE predictions (
    prediction_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name              TEXT NOT NULL,
    target_draw_no          INTEGER,
    predicted_numbers_json  TEXT NOT NULL,
    confidence_json         TEXT,
    created_at              TEXT DEFAULT (datetime('now'))
);
''')
    conn.commit()
    print('Database and tables created.')
finally:
    conn.close()
