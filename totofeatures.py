"""totofeatures.py — Refresh pre-computed feature tables.

Run after totoscrape.py to keep number_stats, number_pairs, and ml_features
up to date.

Usage:
    python3 totofeatures.py          # refresh all features
    python3 totofeatures.py test     # run unit tests
"""
import sqlite3, json, logging, unittest
from collections import defaultdict
from itertools import combinations

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def compute_number_stats(cur):
    """Recompute number_stats from jackpot_no (normal numbers only).

    Returns a dict: {number: {'appearances': int, 'last_draw_no': int,
                               'avg_gap': float, 'max_gap': int}}
    """
    cur.execute("""
        SELECT j.number, d.draw_no
        FROM jackpot_no j
        JOIN draws d ON d.draw_no = j.draw_no
        WHERE j.no_type = 'normal'
        ORDER BY j.number, d.draw_no
    """)
    rows = cur.fetchall()

    history = defaultdict(list)
    for number, draw_no in rows:
        history[number].append(draw_no)

    stats = {}
    for number, draw_nos in history.items():
        gaps = [draw_nos[i] - draw_nos[i - 1] for i in range(1, len(draw_nos))]
        stats[number] = {
            'appearances': len(draw_nos),
            'last_draw_no': draw_nos[-1],
            'avg_gap': sum(gaps) / len(gaps) if gaps else None,
            'max_gap': max(gaps) if gaps else None,
        }
    return stats


def compute_number_pairs(cur):
    """Recompute co-occurrence counts for every pair of winning numbers.

    Returns a dict: {(a, b): count}  where a < b.
    """
    cur.execute("""
        SELECT draw_no, number
        FROM jackpot_no
        WHERE no_type = 'normal'
        ORDER BY draw_no
    """)
    draw_numbers = defaultdict(list)
    for draw_no, number in cur.fetchall():
        draw_numbers[draw_no].append(number)

    pair_counts = defaultdict(int)
    for numbers in draw_numbers.values():
        for a, b in combinations(sorted(numbers), 2):
            pair_counts[(a, b)] += 1
    return pair_counts


def compute_ml_features(cur, draw_no, stats, pair_counts):
    """Build a feature snapshot for a single draw (using only prior data).

    Returns a dict suitable for JSON serialisation.
    """
    cur.execute("""
        SELECT number FROM jackpot_no
        WHERE draw_no = ? AND no_type = 'normal'
    """, (draw_no,))
    numbers = [r[0] for r in cur.fetchall()]

    cur.execute("SELECT day FROM draws WHERE draw_no = ?", (draw_no,))
    row = cur.fetchone()
    day_of_week = row[0] if row else None

    cur.execute("""
        SELECT prize_amount FROM draw_prizes
        WHERE draw_no = ? AND prize_group = 1
    """, (draw_no,))
    prize_row = cur.fetchone()
    jackpot_cents = prize_row[0] if prize_row else None

    freqs = [stats.get(n, {}).get('appearances', 0) for n in numbers]
    gaps = [stats.get(n, {}).get('avg_gap') for n in numbers]
    gaps_clean = [g for g in gaps if g is not None]
    pair_density = sum(
        pair_counts.get((min(a, b), max(a, b)), 0)
        for a, b in combinations(numbers, 2)
    )

    return {
        'draw_no': draw_no,
        'numbers': numbers,
        'day_of_week': day_of_week,
        'jackpot_cents': jackpot_cents,
        'freq_mean': sum(freqs) / len(freqs) if freqs else 0,
        'freq_min': min(freqs) if freqs else 0,
        'freq_max': max(freqs) if freqs else 0,
        'gap_mean': sum(gaps_clean) / len(gaps_clean) if gaps_clean else None,
        'gap_max': max(gaps_clean) if gaps_clean else None,
        'pair_density': pair_density,
    }


def refresh_all(db_path='toto.sqlite'):
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()

        logging.info('Computing number stats...')
        stats = compute_number_stats(cur)
        cur.execute('DELETE FROM number_stats')
        for number, s in stats.items():
            cur.execute(
                'INSERT INTO number_stats(number,appearances,last_draw_no,avg_gap,max_gap) VALUES(?,?,?,?,?)',
                (number, s['appearances'], s['last_draw_no'], s['avg_gap'], s['max_gap']),
            )
        logging.info('  %d numbers updated', len(stats))

        logging.info('Computing number pairs...')
        pair_counts = compute_number_pairs(cur)
        cur.execute('DELETE FROM number_pairs')
        for (a, b), count in pair_counts.items():
            cur.execute(
                'INSERT INTO number_pairs(number_a,number_b,co_appearances) VALUES(?,?,?)',
                (a, b, count),
            )
        logging.info('  %d pairs updated', len(pair_counts))

        logging.info('Computing ML features per draw...')
        cur.execute('SELECT draw_no FROM draws WHERE scanned=1 ORDER BY draw_no')
        draw_nos = [r[0] for r in cur.fetchall()]
        cur.execute('DELETE FROM ml_features')
        for draw_no in draw_nos:
            features = compute_ml_features(cur, draw_no, stats, pair_counts)
            cur.execute(
                'INSERT OR REPLACE INTO ml_features(draw_no, features_json) VALUES(?,?)',
                (draw_no, json.dumps(features)),
            )
        logging.info('  %d draw feature snapshots updated', len(draw_nos))

        conn.commit()
        logging.info('Feature refresh complete.')
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestComputeNumberStats(unittest.TestCase):
    def _make_cur(self, rows):
        conn = sqlite3.connect(':memory:')
        cur = conn.cursor()
        cur.executescript('''
            CREATE TABLE draws(draw_no INTEGER PRIMARY KEY, day TEXT, date TEXT, scanned INTEGER DEFAULT 0);
            CREATE TABLE jackpot_no(draw_no INTEGER, no_type TEXT, number INTEGER);
        ''')
        for draw_no in range(1, 11):
            cur.execute('INSERT INTO draws VALUES(?,?,?,1)', (draw_no, 'Mon', f'0{draw_no} Jan 2020'))
        for row in rows:
            cur.execute('INSERT INTO jackpot_no VALUES(?,?,?)', row)
        return cur

    def test_appearance_count(self):
        cur = self._make_cur([(1, 'normal', 7), (2, 'normal', 7), (3, 'normal', 7)])
        stats = compute_number_stats(cur)
        self.assertEqual(stats[7]['appearances'], 3)

    def test_last_draw_no(self):
        cur = self._make_cur([(1, 'normal', 7), (3, 'normal', 7)])
        stats = compute_number_stats(cur)
        self.assertEqual(stats[7]['last_draw_no'], 3)

    def test_avg_gap(self):
        cur = self._make_cur([(1, 'normal', 7), (3, 'normal', 7), (6, 'normal', 7)])
        stats = compute_number_stats(cur)
        self.assertAlmostEqual(stats[7]['avg_gap'], 2.5)

    def test_max_gap(self):
        cur = self._make_cur([(1, 'normal', 7), (4, 'normal', 7), (6, 'normal', 7)])
        stats = compute_number_stats(cur)
        self.assertEqual(stats[7]['max_gap'], 3)

    def test_additional_numbers_excluded(self):
        cur = self._make_cur([(1, 'additional', 7), (2, 'additional', 7)])
        stats = compute_number_stats(cur)
        self.assertNotIn(7, stats)


class TestComputeNumberPairs(unittest.TestCase):
    def _make_cur(self, draws):
        conn = sqlite3.connect(':memory:')
        cur = conn.cursor()
        cur.execute('CREATE TABLE jackpot_no(draw_no INTEGER, no_type TEXT, number INTEGER)')
        for draw_no, numbers in draws.items():
            for n in numbers:
                cur.execute('INSERT INTO jackpot_no VALUES(?,?,?)', (draw_no, 'normal', n))
        return cur

    def test_pair_count(self):
        cur = self._make_cur({1: [1, 2, 3], 2: [1, 2, 4]})
        pairs = compute_number_pairs(cur)
        self.assertEqual(pairs[(1, 2)], 2)
        self.assertEqual(pairs[(1, 3)], 1)
        self.assertEqual(pairs.get((2, 4), 0), 1)

    def test_key_order_a_less_than_b(self):
        cur = self._make_cur({1: [5, 3]})
        pairs = compute_number_pairs(cur)
        self.assertIn((3, 5), pairs)
        self.assertNotIn((5, 3), pairs)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        sys.argv.pop(1)
        unittest.main()
    else:
        refresh_all()
