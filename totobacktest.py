"""totobacktest.py — Strategy backtesting engine.

Replays a pick strategy against every historical draw in chronological order.
No lookahead: features for draw N are computed from draws 1…N-1 only.

Strategies (strategy_json):
  top_frequency  — pick the N numbers with highest appearance count to date
  top_pair       — pick numbers forming the most frequent pairs to date
  custom         — provide a fixed list of numbers

Usage:
    python3 totobacktest.py --strategy top_frequency --window 20 --name "Freq-20"
    python3 totobacktest.py --strategy custom --numbers 7,14,21,28,35,42 --name "Lucky7s"
    python3 totobacktest.py --list
    python3 totobacktest.py --results <backtest_id>
    python3 totobacktest.py test
"""
import sqlite3, json, logging, argparse, unittest
from collections import defaultdict
from itertools import combinations
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

PRIZE_GROUPS = {6: 1, 5: 2, 4: 3, 3: 4}   # matched_count -> prize_group (approx)


def pick_top_frequency(draw_history, n=6, window=None):
    """Pick the n most frequently drawn numbers in the last `window` draws."""
    if window:
        draw_history = draw_history[-window:]
    freq = defaultdict(int)
    for numbers in draw_history:
        for num in numbers:
            freq[num] += 1
    return sorted(freq, key=lambda x: -freq[x])[:n]


def pick_top_pair(draw_history, n=6, window=None):
    """Pick numbers involved in the most frequent pairs."""
    if window:
        draw_history = draw_history[-window:]
    pair_counts = defaultdict(int)
    for numbers in draw_history:
        for a, b in combinations(sorted(numbers), 2):
            pair_counts[(a, b)] += 1
    node_score = defaultdict(int)
    for (a, b), count in pair_counts.items():
        node_score[a] += count
        node_score[b] += count
    return sorted(node_score, key=lambda x: -node_score[x])[:n]


def score_pick(picked, actual_normal, actual_additional):
    """Return (matched_count, prize_group).

    Prize groups (simplified, no additional number bonus included):
      6 matched  → Group 1
      5 matched  → Group 2
      4 matched  → Group 3
      3 matched  → Group 4
      else       → no prize (group 0)
    """
    matched = len(set(picked) & set(actual_normal))
    prize_group = PRIZE_GROUPS.get(matched, 0)
    return matched, prize_group


def run_backtest(cur, strategy_json, backtest_id):
    """Replay a strategy against all scanned draws. Write to backtest_results."""
    strategy = json.loads(strategy_json)
    method = strategy.get('pick_method', 'top_frequency')
    window = strategy.get('window_draws')
    fixed_numbers = strategy.get('numbers')

    cur.execute('SELECT draw_no FROM draws WHERE scanned=1 ORDER BY draw_no')
    draw_nos = [r[0] for r in cur.fetchall()]

    draw_history = []   # list of number lists, grows as we advance
    wins = 0

    for draw_no in draw_nos:
        # pick using only history so far (no lookahead)
        if method == 'top_frequency':
            picked = pick_top_frequency(draw_history, n=6, window=window)
        elif method == 'top_pair':
            picked = pick_top_pair(draw_history, n=6, window=window)
        elif method == 'custom':
            picked = list(fixed_numbers)[:6]
        else:
            picked = []

        # fetch actual result for this draw
        cur.execute(
            'SELECT number FROM jackpot_no WHERE draw_no=? AND no_type="normal"',
            (draw_no,),
        )
        actual_normal = [r[0] for r in cur.fetchall()]
        cur.execute(
            'SELECT number FROM jackpot_no WHERE draw_no=? AND no_type="additional"',
            (draw_no,),
        )
        actual_add_row = cur.fetchone()
        actual_additional = [actual_add_row[0]] if actual_add_row else []

        matched_count, prize_group = score_pick(picked, actual_normal, actual_additional)

        # look up prize amount from draw_prizes
        prize_amount = None
        if prize_group > 0:
            cur.execute(
                'SELECT prize_amount FROM draw_prizes WHERE draw_no=? AND prize_group=?',
                (draw_no, prize_group),
            )
            prize_row = cur.fetchone()
            prize_amount = prize_row[0] if prize_row else None
            wins += 1

        cur.execute(
            """INSERT OR REPLACE INTO backtest_results
               (backtest_id,draw_no,picked_numbers_json,matched_count,prize_group,prize_amount)
               VALUES(?,?,?,?,?,?)""",
            (backtest_id, draw_no, json.dumps(sorted(picked)),
             matched_count, prize_group, prize_amount),
        )

        # update history for next draw
        draw_history.append(actual_normal)

    return wins, len(draw_nos)


def print_results(cur, backtest_id):
    cur.execute('SELECT name, strategy_json FROM backtests WHERE backtest_id=?', (backtest_id,))
    row = cur.fetchone()
    if not row:
        print('Backtest not found.')
        return
    name, strategy_json = row
    print(f'\nBacktest [{backtest_id}]: {name}')
    print(f'Strategy: {strategy_json}\n')

    # --- aggregate by prize group ---
    cur.execute("""
        SELECT prize_group, COUNT(*) as cnt,
               SUM(CASE WHEN prize_amount IS NOT NULL THEN prize_amount ELSE 0 END) as total_prize
        FROM backtest_results
        WHERE backtest_id=?
        GROUP BY prize_group
        ORDER BY prize_group
    """, (backtest_id,))
    rows = cur.fetchall()
    total_draws = sum(r[1] for r in rows)
    group_map = {r[0]: r for r in rows}

    print(f'{"Group":<12} {"Draws":<8} {"Total Prize ($)":<20}')
    print('-' * 42)
    for prize_group, cnt, total_prize in rows:
        label = f'Group {prize_group}' if prize_group > 0 else 'No prize'
        prize_str = f'{total_prize / 100:,.2f}' if total_prize else '-'
        print(f'{label:<12} {cnt:<8} {prize_str:<20}')
    print(f'\nTotal draws replayed: {total_draws}')

    # --- perfect-match (Group 1 / 6-of-6) highlight ---
    perfect = group_map.get(1)
    if perfect and perfect[1] > 0:
        print(f'\n*** PERFECT MATCH(ES) FOUND: {perfect[1]} draw(s) with all 6 numbers correct ***')
        cur.execute("""
            SELECT draw_no, picked_numbers_json
            FROM backtest_results
            WHERE backtest_id=? AND prize_group=1
            ORDER BY draw_no
        """, (backtest_id,))
        for draw_no, picked_json in cur.fetchall():
            print(f'  Draw {draw_no}: picked {json.loads(picked_json)}')
    else:
        # show best result achieved even without a perfect match
        cur.execute("""
            SELECT MAX(matched_count), draw_no, picked_numbers_json
            FROM backtest_results
            WHERE backtest_id=?
        """, (backtest_id,))
        best = cur.fetchone()
        if best and best[0]:
            print(f'\nBest result: {best[0]}/6 matched on draw {best[1]}  picked={json.loads(best[2])}')

    # --- hit-rate summary (3+ matches = "near miss or better") ---
    near = sum(r[1] for r in rows if r[0] > 0)   # any prize group
    if total_draws:
        print(f'Hit rate (any prize): {near}/{total_draws} = {100 * near / total_draws:.1f}%')


def main():
    parser = argparse.ArgumentParser(description='TOTO backtesting engine')
    parser.add_argument('--strategy', choices=['top_frequency', 'top_pair', 'custom'])
    parser.add_argument('--window',  type=int, default=None)
    parser.add_argument('--numbers', help='Comma-separated for custom strategy')
    parser.add_argument('--name',    default='Unnamed')
    parser.add_argument('--list',    action='store_true')
    parser.add_argument('--results', type=int, metavar='ID')
    args = parser.parse_args()

    conn = sqlite3.connect('toto.sqlite')
    try:
        cur = conn.cursor()

        if args.list:
            cur.execute('SELECT backtest_id, name, created_at FROM backtests ORDER BY backtest_id')
            for bid, name, created_at in cur.fetchall():
                print(f'[{bid}] {name}  ({created_at})')

        elif args.results is not None:
            print_results(cur, args.results)

        elif args.strategy:
            strategy = {'pick_method': args.strategy}
            if args.window:
                strategy['window_draws'] = args.window
            if args.strategy == 'custom':
                if not args.numbers:
                    parser.error('--numbers required for custom strategy')
                strategy['numbers'] = [int(x) for x in args.numbers.split(',')]

            strategy_json = json.dumps(strategy)
            cur.execute(
                'INSERT INTO backtests(name,strategy_json,created_at) VALUES(?,?,?)',
                (args.name, strategy_json, datetime.utcnow().isoformat()),
            )
            backtest_id = cur.lastrowid
            logging.info('Running backtest ID %d...', backtest_id)
            wins, total = run_backtest(cur, strategy_json, backtest_id)
            conn.commit()
            logging.info('Done: %d wins in %d draws (%.1f%%)', wins, total,
                         100 * wins / total if total else 0)
            print_results(cur, backtest_id)

        else:
            parser.print_help()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestPickTopFrequency(unittest.TestCase):
    def test_picks_most_frequent(self):
        history = [[1, 2, 3], [1, 2, 4], [1, 5, 6]]
        picked = pick_top_frequency(history, n=2)
        self.assertEqual(picked[0], 1)   # appears 3 times
        self.assertIn(2, picked)

    def test_window_limits_history(self):
        history = [[99, 1, 2], [1, 2, 3], [1, 2, 3]]
        picked = pick_top_frequency(history, n=3, window=2)
        self.assertNotIn(99, picked)

    def test_returns_n_numbers(self):
        history = [[i, i+1, i+2, i+3, i+4, i+5] for i in range(10)]
        self.assertEqual(len(pick_top_frequency(history, n=6)), 6)


class TestPickTopPair(unittest.TestCase):
    def test_picks_highly_connected_numbers(self):
        # 7 co-appears with 3 different numbers; 99 only appears once
        history = [[7, 14, 1, 2, 3, 4], [7, 21, 5, 6, 8, 9], [7, 28, 10, 11, 12, 13], [99, 1, 2, 3, 4, 5]]
        picked = pick_top_pair(history, n=1)
        self.assertEqual(picked[0], 7)


class TestScorePick(unittest.TestCase):
    def test_six_matched_group1(self):
        matched, group = score_pick([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [])
        self.assertEqual(matched, 6)
        self.assertEqual(group, 1)

    def test_three_matched_group4(self):
        matched, group = score_pick([1, 2, 3, 7, 8, 9], [1, 2, 3, 4, 5, 6], [])
        self.assertEqual(matched, 3)
        self.assertEqual(group, 4)

    def test_two_matched_no_prize(self):
        matched, group = score_pick([1, 2, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6], [])
        self.assertEqual(matched, 2)
        self.assertEqual(group, 0)


class TestRunBacktest(unittest.TestCase):
    def _make_db(self):
        conn = sqlite3.connect(':memory:')
        cur = conn.cursor()
        cur.executescript('''
            CREATE TABLE draws(draw_no INTEGER PRIMARY KEY, day TEXT, date TEXT,
                               draw_type TEXT DEFAULT "ordinary", scanned INTEGER DEFAULT 0);
            CREATE TABLE jackpot_no(draw_no INTEGER, no_type TEXT, number INTEGER);
            CREATE TABLE draw_prizes(draw_no INTEGER, prize_group INTEGER,
                                     prize_amount INTEGER, winner_count INTEGER);
            CREATE TABLE backtests(backtest_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                   name TEXT, strategy_json TEXT, created_at TEXT);
            CREATE TABLE backtest_results(backtest_id INTEGER, draw_no INTEGER,
                picked_numbers_json TEXT, matched_count INTEGER,
                prize_group INTEGER, prize_amount INTEGER,
                PRIMARY KEY(backtest_id, draw_no));
        ''')
        # 5 draws, always same winning numbers
        for draw_no in range(1, 6):
            cur.execute('INSERT INTO draws VALUES(?,?,?,?,1)', (draw_no, 'Mon', f'0{draw_no} Jan 2024', 'ordinary'))
            for n in [1, 2, 3, 4, 5, 6]:
                cur.execute('INSERT INTO jackpot_no VALUES(?,?,?)', (draw_no, 'normal', n))
            cur.execute('INSERT INTO jackpot_no VALUES(?,?,?)', (draw_no, 'additional', 42))
        return conn, cur

    def test_custom_strategy_all_wins(self):
        conn, cur = self._make_db()
        cur.execute('INSERT INTO backtests VALUES(1,"test","{}","2024-01-01")')
        strategy = json.dumps({'pick_method': 'custom', 'numbers': [1, 2, 3, 4, 5, 6]})
        wins, total = run_backtest(cur, strategy, 1)
        self.assertEqual(total, 5)
        self.assertEqual(wins, 5)   # all 6 matched → Group 1 every draw

    def test_custom_strategy_no_wins(self):
        conn, cur = self._make_db()
        cur.execute('INSERT INTO backtests VALUES(1,"test","{}","2024-01-01")')
        strategy = json.dumps({'pick_method': 'custom', 'numbers': [7, 8, 9, 10, 11, 12]})
        wins, total = run_backtest(cur, strategy, 1)
        self.assertEqual(wins, 0)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        sys.argv.pop(1)
        unittest.main()
    else:
        main()
