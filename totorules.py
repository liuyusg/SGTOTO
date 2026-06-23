"""totorules.py — Custom rule engine for finding draw patterns.

Rules are stored as JSON in the 'rules' table and evaluated against
historical draw data. Results are cached in 'rule_results'.

Supported rule types:
  frequency  — numbers that appeared at least N times in last W draws
  pair       — a specific pair that co-appeared
  gap        — numbers whose gap before appearing exceeded a threshold
  temporal   — numbers correlated with draw date features (month, day-of-month,
               previous/next draw dates); discovered via frequency analysis and
               decision trees (sklearn optional)
  lag         — numbers correlated with what appeared in T-1, T-2, T-3 draws;
               transition pairs and recency-weighted clusters (T-1 weight 3,
               T-2 weight 2, T-3 weight 1)
  correlation — numbers elevated after a draw whose pairs sum to S or differ by D;
               consecutive variant: condition must fire in both T-1 and T-2

Usage:
    python3 totorules.py --add   '{"name":"Hot pair","type":"pair","number_a":7,"number_b":14}'
    python3 totorules.py --list
    python3 totorules.py --run <rule_id>
    python3 totorules.py --run all
    python3 totorules.py --generate          # auto-generate all rule types
    python3 totorules.py --predict            # score numbers for the next draw
    python3 totorules.py --favourites 1,15,22,8,28,35   # save favourite numbers
    python3 totorules.py --predict --focus    # show only favourite numbers
    python3 totorules.py test
"""
import sqlite3, json, logging, argparse, os, unittest
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from itertools import combinations

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

FAVOURITES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'favourites.json')

MONTH_ABBR = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
}
# Mon draw → next TOTO is Thu (+3 days); Thu draw → next TOTO is Mon (+4 days)
_DOW_NEXT_DAYS = {'Mon': 3, 'Thu': 4}

# Number buckets: index 0-4
BUCKET_RANGES = [(1, 9), (10, 19), (20, 29), (30, 39), (40, 49)]
BUCKET_LABELS = ['1-9', '10-19', '20-29', '30-39', '40-49']


def _bucket_dist(nums):
    """Return tuple (c0,c1,c2,c3,c4) — count per bucket for a list of numbers."""
    counts = [0] * 5
    for n in nums:
        for i, (lo, hi) in enumerate(BUCKET_RANGES):
            if lo <= n <= hi:
                counts[i] += 1
                break
    return tuple(counts)


def _odd_count(nums):
    """Return how many numbers in nums are odd."""
    return sum(1 for n in nums if n % 2 == 1)


# Seed-number derivation methods.
# Works for any seed value (single-digit 1-9 by default, or extra configured seeds).
# Each function returns the set of 1-49 numbers "derivable" from the seed.
SEED_DERIVATIONS = {
    # All multiples of seed within 1-49  (e.g. seed=7 → 7,14,21,28,35,42,49)
    'multiple':   lambda s: frozenset(s * k for k in range(1, 50 // s + 1) if s * k <= 49),
    # Numbers sharing the same last digit as seed  (e.g. seed=13 → 3,13,23,33,43)
    'last_digit': lambda s: frozenset(n for n in range(1, 50) if n % 10 == s % 10),
    # Numbers whose digit-sum equals the digit-sum of seed
    # (e.g. seed=2 → 2,11,20,29,38,47;  seed=13 → digit_sum=4 → 4,13,22,31,40,49)
    'digit_sum':  lambda s: frozenset(
        n for n in range(1, 50)
        if sum(int(d) for d in str(n)) == sum(int(d) for d in str(s))
    ),
}

DEFAULT_SEEDS = list(range(1, 10))   # 1-9
SEEDS_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'seeds.json')


def load_seeds():
    """Return the active seed list: default 1-9 plus any extras in seeds.json."""
    base = list(range(1, 10))
    try:
        with open(SEEDS_FILE) as f:
            data = json.load(f)
        extra = sorted(set(int(n) for n in data.get('extra', []) if 1 <= int(n) <= 49))
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        extra = []
    return sorted(set(base + extra))


def save_seeds(extra_numbers):
    """Persist extra seed numbers (beyond 1-9) to seeds.json."""
    extra = sorted(set(int(n) for n in extra_numbers if 1 <= int(n) <= 49 and int(n) not in range(1, 10)))
    with open(SEEDS_FILE, 'w') as f:
        json.dump({
            'default': list(range(1, 10)),
            'extra':   extra,
            'active':  sorted(set(range(1, 10)) | set(extra)),
            'updated': datetime.utcnow().isoformat(),
        }, f, indent=2)
    return sorted(set(range(1, 10)) | set(extra))


def _seed_coverage(nums, seed, method):
    """Return frozenset of nums that are 'derived from' seed via method."""
    derived = SEED_DERIVATIONS[method](seed)
    return frozenset(n for n in nums if n in derived)


# ---------------------------------------------------------------------------
# Favourite numbers  (persisted in favourites.json alongside this script)
# ---------------------------------------------------------------------------

def load_favourites():
    """Load favourite numbers from favourites.json. Returns [] if not set."""
    try:
        with open(FAVOURITES_FILE) as f:
            data = json.load(f)
        return sorted(set(int(n) for n in data.get('numbers', []) if 1 <= int(n) <= 49))
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return []


def save_favourites(numbers):
    """Validate and persist favourite numbers to favourites.json."""
    nums = sorted(set(int(n) for n in numbers if 1 <= int(n) <= 49))
    if not nums:
        raise ValueError('No valid numbers in 1-49 provided.')
    with open(FAVOURITES_FILE, 'w') as f:
        json.dump({'numbers': nums, 'updated': datetime.utcnow().isoformat()}, f, indent=2)
    return nums


def print_favourite_analysis(cur, favourites, ranked, rule_details):
    """Print a focused prediction report for the configured favourite numbers.

    For each favourite:
      - Its overall rank and score among all 49 numbers
      - Every rule currently firing that supports it
      - Up to 3 dormant rules that historically predict it (condition not met now)
    """
    if not favourites:
        print('No favourites configured. Use --favourites N1,N2,... to set them.')
        return

    score_map  = dict(ranked)
    all_scores = sorted({s for _, s in ranked}, reverse=True)

    def _rank(n):
        s = score_map.get(n, 0)
        if s == 0:
            return None
        for i, sv in enumerate(all_scores, 1):
            if sv == s:
                return i
        return None

    # Build set of rule names that ARE firing (contributed to score)
    firing_rule_names = set()
    for rules in rule_details.values():
        for r in rules:
            firing_rule_names.add(r.split(' (+')[0])

    # Index: rule_name -> rule dict, for dormant lookup
    cur.execute('SELECT name, rule_json FROM rules')
    all_rule_rows = cur.fetchall()
    rules_by_name = {name: json.loads(rj) for name, rj in all_rule_rows}

    fav_by_score = sorted(favourites, key=lambda n: -score_map.get(n, 0))
    top6_set     = {n for n, _ in ranked[:6]}

    print()
    print('=' * 70)
    print(f'FAVOURITE NUMBERS ANALYSIS')
    print(f'Configured: {favourites}')
    print('=' * 70)

    for fav in fav_by_score:
        score       = score_map.get(fav, 0.0)
        rank_pos    = _rank(fav)
        firing      = rule_details.get(fav, [])
        star        = '\u2605' if fav in top6_set else ' '
        rank_str    = f'rank #{rank_pos}' if rank_pos else 'unranked'

        print(f'\n{star} {fav:>3}   score={score:.2f}  {rank_str}')

        if firing:
            print('  Firing rules:')
            for r in firing[:6]:
                print(f'    \u2713 {r[:68]}')
        else:
            print('  No rules currently fire for this number.')

        # Dormant: rules that list fav in numbers[] but are not firing
        dormant = [
            name for name, rule in rules_by_name.items()
            if fav in rule.get('numbers', []) and name not in firing_rule_names
        ]
        if dormant:
            print(f'  Dormant rules (historically predict {fav}, condition not met now):')
            for d in dormant[:3]:
                print(f'    \u25cb {d[:68]}')

    overlap = sorted(n for n in favourites if n in top6_set)
    print()
    print(f'Summary: {len(overlap)}/{len(favourites)} favourites in top-6  →  {overlap}')
    print('=' * 70)


# ---------------------------------------------------------------------------
# Temporal feature helpers
# ---------------------------------------------------------------------------

def _parse_date_features(date_str, day_str=None, prev_date_str=None, next_date_str=None):
    """Return a temporal feature dict for a draw date string.

    date_str     : '08 Apr 2024'  (DD Mon YYYY)
    day_str      : 'Mon' or 'Thu' — used to compute the next TOTO draw date
    prev_date_str: date_str of the immediately preceding draw
    next_date_str: date_str of the immediately following draw (overrides computed)

    Produced keys (all integers; absent when data is unavailable):
      dom, month           — current draw date
      prev_dom, prev_month — preceding draw date
      next_dom, next_month — following draw date
    """
    def _unpack(s):
        p = s.strip().split()
        return int(p[0]), MONTH_ABBR.get(p[1], 0), int(p[2]) if len(p) > 2 else 0

    dom, month, year = _unpack(date_str)
    features = {'dom': dom, 'month': month}

    # Compute next TOTO draw date from the current draw's day-of-week
    days_ahead = _DOW_NEXT_DAYS.get(day_str) if day_str else None
    if days_ahead and year:
        try:
            ndt = datetime(year, month, dom) + timedelta(days=days_ahead)
            features['next_dom'] = ndt.day
            features['next_month'] = ndt.month
        except ValueError:
            pass

    if prev_date_str:
        p_dom, p_month, _ = _unpack(prev_date_str)
        features['prev_dom'] = p_dom
        features['prev_month'] = p_month

    if next_date_str:          # actual next draw overrides the computed value
        n_dom, n_month, _ = _unpack(next_date_str)
        features['next_dom'] = n_dom
        features['next_month'] = n_month

    return features


def eval_temporal_rule(cur, draw_no, rule):
    """Evaluate a temporal rule against draw_no.

    Rule JSON fields:
      conditions  : list of {feature, op, value}  (op: eq / in / gte / lte)
      numbers     : list of integers expected among winning numbers
      min_matches : minimum count of 'numbers' that must appear (default 1)
    """
    cur.execute('SELECT day, date FROM draws WHERE draw_no=?', (draw_no,))
    row = cur.fetchone()
    if not row or not row[1]:
        return False, {}
    day_str, date_str = row

    cur.execute('SELECT date FROM draws WHERE draw_no < ? AND scanned=1 ORDER BY draw_no DESC LIMIT 1',
                (draw_no,))
    prev_row = cur.fetchone()
    cur.execute('SELECT date FROM draws WHERE draw_no > ? AND scanned=1 ORDER BY draw_no ASC LIMIT 1',
                (draw_no,))
    next_row = cur.fetchone()

    features = _parse_date_features(
        date_str, day_str,
        prev_date_str=prev_row[0] if prev_row else None,
        next_date_str=next_row[0] if next_row else None,
    )

    for cond in rule.get('conditions', []):
        fv  = features.get(cond['feature'])
        op  = cond.get('op', 'eq')
        val = cond['value']
        if fv is None:                       return False, {}
        if   op == 'eq'  and fv != val:      return False, {}
        elif op == 'in'  and fv not in val:  return False, {}
        elif op == 'gte' and fv < val:       return False, {}
        elif op == 'lte' and fv > val:       return False, {}

    numbers     = rule.get('numbers', [])
    min_matches = rule.get('min_matches', 1)
    cur.execute('SELECT number FROM jackpot_no WHERE draw_no=? AND no_type="normal"', (draw_no,))
    actual  = {int(r[0]) for r in cur.fetchall()}
    matched = [n for n in numbers if n in actual]
    return len(matched) >= min_matches, {'features': features, 'matched_numbers': matched}


def eval_lag_rule(cur, draw_no, rule):
    """Evaluate a lag rule against draw_no.

    Rule JSON fields:
      lag_conditions : list of {lag: 1|2|3, numbers: [...], min_present: N}
                       'min_present' numbers must have appeared in T-lag draw.
                       Use min_present=0 for an absence condition.
      numbers        : predicted numbers for the current draw
      min_matches    : how many 'numbers' must appear (default 1)
    """
    cur.execute(
        'SELECT draw_no FROM draws WHERE draw_no < ? AND scanned=1 ORDER BY draw_no DESC LIMIT 3',
        (draw_no,),
    )
    prev_nos = [r[0] for r in cur.fetchall()]   # [T-1, T-2, T-3] most-recent first

    lag = {}   # lag_k (1/2/3) -> frozenset of numbers in that draw
    for k, prev in enumerate(prev_nos, 1):
        cur.execute(
            'SELECT number FROM jackpot_no WHERE draw_no=? AND no_type="normal"',
            (prev,),
        )
        lag[k] = frozenset(int(r[0]) for r in cur.fetchall())

    for cond in rule.get('lag_conditions', []):
        k           = cond['lag']
        needed      = set(cond.get('numbers', []))
        min_present = cond.get('min_present', 1)
        if k not in lag:
            return False, {}
        present = len(needed & lag[k])
        if present < min_present:
            return False, {}

    cur.execute(
        'SELECT number FROM jackpot_no WHERE draw_no=? AND no_type="normal"',
        (draw_no,),
    )
    actual      = {int(r[0]) for r in cur.fetchall()}
    numbers     = rule.get('numbers', [])
    min_matches = rule.get('min_matches', 1)
    matched     = [n for n in numbers if n in actual]
    lag_info    = {k: sorted(v) for k, v in lag.items()}
    return len(matched) >= min_matches, {'lag': lag_info, 'matched_numbers': matched}


# ---------------------------------------------------------------------------
# Rule evaluation
# ---------------------------------------------------------------------------

def eval_frequency_rule(cur, draw_no, rule):
    """Does this draw contain at least min_matches of the specified numbers?"""
    numbers = rule.get('numbers', [])
    min_matches = rule.get('min_matches', 1)
    cur.execute(
        'SELECT number FROM jackpot_no WHERE draw_no=? AND no_type="normal"',
        (draw_no,),
    )
    actual = {r[0] for r in cur.fetchall()}
    matched = [n for n in numbers if n in actual]
    return len(matched) >= min_matches, {'matched_numbers': matched}


def eval_pair_rule(cur, draw_no, rule):
    """Did this draw contain both numbers in the pair?"""
    a, b = rule['number_a'], rule['number_b']
    cur.execute(
        'SELECT COUNT(*) FROM jackpot_no WHERE draw_no=? AND no_type="normal" AND number IN (?,?)',
        (draw_no, a, b),
    )
    count = cur.fetchone()[0]
    matched = count == 2
    return matched, {'pair': [a, b], 'both_present': matched}


def eval_gap_rule(cur, draw_no, rule):
    """Did any winning number have a gap >= min_gap before this draw?"""
    min_gap = rule.get('min_gap', 10)
    cur.execute(
        'SELECT number FROM jackpot_no WHERE draw_no=? AND no_type="normal"',
        (draw_no,),
    )
    winning = [r[0] for r in cur.fetchall()]
    overdue = []
    for number in winning:
        cur.execute(
            """SELECT draw_no FROM jackpot_no
               WHERE no_type='normal' AND number=? AND draw_no < ?
               ORDER BY draw_no DESC LIMIT 1""",
            (number, draw_no),
        )
        prev = cur.fetchone()
        if prev:
            gap = draw_no - prev[0]
            if gap >= min_gap:
                overdue.append({'number': number, 'gap': gap})
    return bool(overdue), {'overdue_numbers': overdue}


def eval_correlation_rule(cur, draw_no, rule):
    """Evaluate a correlation rule for draw_no.

    Rule JSON fields:
      cond_type   : 'pair_sum' or 'pair_diff'
      cond_value  : the target sum or absolute difference
      consecutive : if True, both T-1 AND T-2 must satisfy the condition
      numbers     : predicted numbers when condition fires
      min_matches : how many 'numbers' must appear in draw_no (default 1)
    """
    cond_type   = rule.get('cond_type')
    cond_value  = rule.get('cond_value')
    consecutive = rule.get('consecutive', False)

    def _nums(dn):
        cur.execute(
            'SELECT number FROM jackpot_no WHERE draw_no=? AND no_type="normal"',
            (dn,)
        )
        return [int(r[0]) for r in cur.fetchall()]

    def _fires(nums):
        if cond_type == 'pair_sum':
            return any(a + b == cond_value for a, b in combinations(nums, 2))
        if cond_type == 'pair_diff':
            return any(abs(a - b) == cond_value for a, b in combinations(nums, 2))
        return False

    cur.execute(
        'SELECT draw_no FROM draws WHERE draw_no<? AND scanned=1 ORDER BY draw_no DESC LIMIT 2',
        (draw_no,)
    )
    prev = [r[0] for r in cur.fetchall()]
    if not prev:
        return False, {}

    if not _fires(_nums(prev[0])):
        return False, {'fired': False}

    if consecutive:
        if len(prev) < 2 or not _fires(_nums(prev[1])):
            return False, {'fired': False, 'consecutive_failed': True}

    actual      = set(_nums(draw_no))
    numbers     = rule.get('numbers', [])
    min_matches = rule.get('min_matches', 1)
    matched     = [n for n in numbers if n in actual]
    return len(matched) >= min_matches, {
        'cond_type': cond_type, 'cond_value': cond_value,
        'consecutive': consecutive, 'matched_numbers': matched,
    }


def eval_bucket_count_rule(cur, draw_no, rule):
    """Rule fires if T-1 draw has exactly `count` numbers in `bucket` (0-based).

    Rule JSON fields:
      bucket   : 0=1-9, 1=10-19, 2=20-29, 3=30-39, 4=40-49
      count    : exact number of T-1 winning numbers that fall in this bucket
      numbers  : predicted numbers when condition fires
    """
    bucket = rule.get('bucket')
    count  = rule.get('count')
    cur.execute(
        'SELECT draw_no FROM draws WHERE draw_no<? AND scanned=1 ORDER BY draw_no DESC LIMIT 1',
        (draw_no,)
    )
    row = cur.fetchone()
    if not row:
        return False, {}
    cur.execute('SELECT number FROM jackpot_no WHERE draw_no=? AND no_type="normal"', (row[0],))
    nums = [int(r[0]) for r in cur.fetchall()]
    lo, hi = BUCKET_RANGES[bucket]
    if sum(1 for n in nums if lo <= n <= hi) != count:
        return False, {'fired': False}
    cur.execute('SELECT number FROM jackpot_no WHERE draw_no=? AND no_type="normal"', (draw_no,))
    actual  = {int(r[0]) for r in cur.fetchall()}
    numbers = rule.get('numbers', [])
    matched = [n for n in numbers if n in actual]
    return len(matched) >= rule.get('min_matches', 1), {
        'bucket': bucket, 'count': count, 'matched_numbers': matched,
    }


def eval_odd_even_rule(cur, draw_no, rule):
    """Rule fires if T-1 draw has exactly `odd_count` odd numbers.

    Rule JSON fields:
      odd_count  : exact count of odd numbers in the T-1 draw (0-6)
      even_count : 6 - odd_count (informational)
      numbers    : predicted numbers when condition fires
    """
    target_odd = rule.get('odd_count')
    cur.execute(
        'SELECT draw_no FROM draws WHERE draw_no<? AND scanned=1 ORDER BY draw_no DESC LIMIT 1',
        (draw_no,)
    )
    row = cur.fetchone()
    if not row:
        return False, {}
    cur.execute('SELECT number FROM jackpot_no WHERE draw_no=? AND no_type="normal"', (row[0],))
    nums = [int(r[0]) for r in cur.fetchall()]
    if _odd_count(nums) != target_odd:
        return False, {'fired': False}
    cur.execute('SELECT number FROM jackpot_no WHERE draw_no=? AND no_type="normal"', (draw_no,))
    actual  = {int(r[0]) for r in cur.fetchall()}
    numbers = rule.get('numbers', [])
    matched = [n for n in numbers if n in actual]
    return len(matched) >= rule.get('min_matches', 1), {
        'odd_count': target_odd, 'matched_numbers': matched,
    }


def eval_seed_rule(cur, draw_no, rule):
    """Rule fires when T-1 contains a seed number (1-9) and at least `min_derived`
    of its numbers are derivable from that seed via the specified method.

    Rule JSON fields:
      seed         : any integer 1-49 (default seeds 1-9, extra seeds from seeds.json)
      method       : 'multiple' | 'last_digit' | 'digit_sum'
      min_derived  : minimum count of T-1 numbers derivable from seed (default 2;
                     the seed itself always counts as 1)
      numbers      : predicted numbers for next draw
    """
    seed        = rule.get('seed')
    method      = rule.get('method')
    min_derived = rule.get('min_derived', 2)
    if not isinstance(seed, int) or not (1 <= seed <= 49) or method not in SEED_DERIVATIONS:
        return False, {}
    cur.execute(
        'SELECT draw_no FROM draws WHERE draw_no<? AND scanned=1 ORDER BY draw_no DESC LIMIT 1',
        (draw_no,)
    )
    row = cur.fetchone()
    if not row:
        return False, {}
    cur.execute('SELECT number FROM jackpot_no WHERE draw_no=? AND no_type="normal"', (row[0],))
    t1_nums = [int(r[0]) for r in cur.fetchall()]
    if seed not in t1_nums:
        return False, {'fired': False}          # seed not present in T-1
    covered = _seed_coverage(t1_nums, seed, method)
    if len(covered) < min_derived:
        return False, {'fired': False}
    cur.execute('SELECT number FROM jackpot_no WHERE draw_no=? AND no_type="normal"', (draw_no,))
    actual  = {int(r[0]) for r in cur.fetchall()}
    numbers = rule.get('numbers', [])
    matched = [n for n in numbers if n in actual]
    return len(matched) >= rule.get('min_matches', 1), {
        'seed': seed, 'method': method, 'covered': sorted(covered),
        'matched_numbers': matched,
    }


RULE_EVALUATORS = {
    'frequency':    eval_frequency_rule,
    'pair':         eval_pair_rule,
    'gap':          eval_gap_rule,
    'temporal':     eval_temporal_rule,
    'lag':          eval_lag_rule,
    'correlation':  eval_correlation_rule,
    'bucket_count': eval_bucket_count_rule,
    'odd_even':     eval_odd_even_rule,
    'seed':         eval_seed_rule,
}


def evaluate_rule(cur, rule_id, rule_json):
    """Evaluate a rule against all scanned draws; write results to rule_results."""
    rule = json.loads(rule_json)
    rule_type = rule.get('type')
    evaluator = RULE_EVALUATORS.get(rule_type)
    if evaluator is None:
        logging.warning('Unknown rule type: %s', rule_type)
        return 0

    cur.execute('SELECT draw_no FROM draws WHERE scanned=1 ORDER BY draw_no')
    draw_nos = [r[0] for r in cur.fetchall()]
    count = 0
    for draw_no in draw_nos:
        matched, details = evaluator(cur, draw_no, rule)
        cur.execute(
            'INSERT OR REPLACE INTO rule_results(rule_id,draw_no,matched,details_json) VALUES(?,?,?,?)',
            (rule_id, draw_no, int(matched), json.dumps(details)),
        )
        if matched:
            count += 1
    return count


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def add_rule(cur, rule_json_str):
    rule = json.loads(rule_json_str)
    name = rule.get('name', 'Unnamed rule')
    description = rule.get('description', '')
    cur.execute(
        'INSERT INTO rules(name,description,rule_json,created_at) VALUES(?,?,?,?)',
        (name, description, rule_json_str, datetime.utcnow().isoformat()),
    )
    return cur.lastrowid


def list_rules(cur):
    cur.execute('SELECT rule_id, name, description, created_at FROM rules ORDER BY rule_id')
    return cur.fetchall()


# ---------------------------------------------------------------------------
# Temporal pattern discovery (AI / statistical)
# ---------------------------------------------------------------------------

def _load_enriched_draws(cur):
    """Return list of (draw_no, features_dict, numbers_frozenset) for all scanned draws.

    ``features`` includes date fields (dom, month, …) and lag number sets
    ``_t1`` / ``_t2`` / ``_t3`` (frozensets of the 1/2/3 preceding draws).
    """
    cur.execute("""
        SELECT d.draw_no, d.day, d.date, GROUP_CONCAT(jn.number, ',')
        FROM draws d
        JOIN jackpot_no jn ON d.draw_no=jn.draw_no AND jn.no_type='normal'
        WHERE d.scanned=1 AND d.date IS NOT NULL
        GROUP BY d.draw_no
        ORDER BY d.draw_no
    """)
    rows = cur.fetchall()

    # Pre-parse all number sets so lag lookups are O(1)
    all_nums = [
        frozenset(int(n) for n in row[3].split(',') if n)
        for row in rows
    ]

    result = []
    for i, (draw_no, day_str, date_str, _) in enumerate(rows):
        prev_date = rows[i - 1][2] if i > 0 else None
        next_date = rows[i + 1][2] if i < len(rows) - 1 else None
        features  = _parse_date_features(date_str, day_str, prev_date, next_date)
        # Attach lag number sets (T-1 has most weight, T-3 least)
        if i >= 1: features['_t1'] = all_nums[i - 1]
        if i >= 2: features['_t2'] = all_nums[i - 2]
        if i >= 3: features['_t3'] = all_nums[i - 3]
        result.append((draw_no, features, all_nums[i]))
    return result


def _discover_by_frequency(enriched, min_lift=1.5, min_draws=5, top_n=6):
    """Return patterns where specific numbers appear at > min_lift × baseline rate.

    Features examined: month, dom, prev_month, next_month, prev_dom, next_dom.
    A pattern is emitted only when at least 2 numbers show elevated lift.
    """
    GLOBAL_RATE = 6.0 / 49
    feat_keys = ['month', 'dom', 'prev_month', 'next_month', 'prev_dom', 'next_dom']
    patterns = []

    for feat_key in feat_keys:
        bucket = defaultdict(list)
        for _, features, numbers in enriched:
            val = features.get(feat_key)
            if val is not None:
                bucket[val].append(numbers)

        for val, num_lists in bucket.items():
            if len(num_lists) < min_draws:
                continue
            n_draws = len(num_lists)
            counts  = Counter(n for nums in num_lists for n in nums)
            hot = sorted(
                [(n, counts[n] / n_draws / GLOBAL_RATE)
                 for n in range(1, 50)
                 if counts[n] / n_draws >= GLOBAL_RATE * min_lift],
                key=lambda x: -x[1],
            )
            if len(hot) >= 2:
                patterns.append({
                    'conditions': [{'feature': feat_key, 'op': 'eq', 'value': val}],
                    'numbers':    [n for n, _ in hot[:top_n]],
                    'lift':       hot[min(top_n, len(hot)) - 1][1],
                    'n_draws':    n_draws,
                    'source':     'frequency',
                })
    return patterns


def _extract_dt_high_lift_leaves(clf, feat_names, global_rate, min_lift, min_draws):
    """Walk a fitted sklearn decision tree; return condition lists for high-lift leaves."""
    try:
        from sklearn.tree import _tree as sk_tree
    except ImportError:
        return []
    tree    = clf.tree_
    results = []
    stack   = [(0, [])]
    while stack:
        node, conds = stack.pop()
        if tree.feature[node] == sk_tree.TREE_UNDEFINED:
            n = tree.n_node_samples[node]
            if n >= min_draws:
                vals     = tree.value[node][0]
                pos_rate = vals[1] / n if len(vals) > 1 else 0.0
                if pos_rate >= global_rate * min_lift:
                    results.append(conds)
        else:
            fn  = feat_names[tree.feature[node]]
            thr = tree.threshold[node]
            stack.append((tree.children_left[node],
                          conds + [{'feature': fn, 'op': 'lte', 'value': int(thr)}]))
            stack.append((tree.children_right[node],
                          conds + [{'feature': fn, 'op': 'gte', 'value': int(thr) + 1}]))
    return results


def _discover_by_decision_tree(enriched, min_lift=1.5, min_draws=5):
    """Train a decision tree per number to find non-linear temporal + lag splits.

    Base features : dom, month, prev_dom, prev_month, next_dom, next_month
    Lag features  : t1_self, t2_self, t3_self  (was THIS number in T-1/T-2/T-3?)

    Numbers sharing identical high-lift leaf conditions are grouped into one rule.
    Returns [] gracefully when scikit-learn is unavailable.
    """
    try:
        from sklearn.tree import DecisionTreeClassifier
        import numpy as np
    except ImportError:
        logging.info('scikit-learn not available; skipping decision tree analysis')
        return []

    GLOBAL_RATE     = 6.0 / 49
    base_feat_names = ['dom', 'month', 'prev_dom', 'prev_month', 'next_dom', 'next_month']
    feat_names      = base_feat_names + ['t1_self', 't2_self', 't3_self']

    # Base feature matrix (same for all numbers)
    X_base = np.array(
        [[f.get(fn, 0) or 0 for fn in base_feat_names] for _, f, _ in enriched],
        dtype=float,
    )

    condition_groups = defaultdict(set)   # conditions_key → set of numbers
    for number in range(1, 50):
        y = np.array([1 if number in nums else 0 for _, _, nums in enriched])
        if y.sum() < max(5, len(enriched) * 0.04):
            continue

        # Per-number lag columns: was this number in T-1 / T-2 / T-3?
        lag_cols = np.array([
            [1 if number in f.get('_t1', frozenset()) else 0,
             1 if number in f.get('_t2', frozenset()) else 0,
             1 if number in f.get('_t3', frozenset()) else 0]
            for _, f, _ in enriched
        ], dtype=float)
        X = np.hstack([X_base, lag_cols])

        clf = DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=max(min_draws, 5), random_state=42)
        clf.fit(X, y)
        for path_conds in _extract_dt_high_lift_leaves(
                clf, feat_names, GLOBAL_RATE, min_lift, min_draws):
            key = tuple(sorted((c['feature'], c['op'], c['value']) for c in path_conds))
            condition_groups[key].add(number)

    patterns = []
    for key, numbers in condition_groups.items():
        if len(numbers) >= 2:
            conds = [{'feature': f, 'op': op, 'value': v} for f, op, v in sorted(key)]
            patterns.append({'conditions': conds, 'numbers': sorted(numbers),
                              'source': 'decision_tree'})
    return patterns


def generate_temporal_rules(cur, min_lift=1.5, min_draws=5, top_n=6):
    """Discover temporal patterns and persist them as 'temporal' rules.

    Runs frequency analysis (always) and decision-tree analysis (sklearn optional).
    Returns list of newly inserted rule IDs.
    """
    enriched = _load_enriched_draws(cur)
    if len(enriched) < min_draws * 2:
        logging.warning('Not enough draws for temporal analysis (%d)', len(enriched))
        return []

    patterns  = _discover_by_frequency(enriched, min_lift, min_draws, top_n)
    patterns += _discover_by_decision_tree(enriched, min_lift, min_draws)

    new_ids = []
    now = datetime.utcnow().isoformat()
    for pat in patterns:
        conds_desc = ', '.join(
            f"{c['feature']}"
            f"{c.get('op','eq').replace('eq','=').replace('gte','>=').replace('lte','<=')}"
            f"{c['value']}"
            for c in pat['conditions']
        )
        name = f"Temporal [{pat['source']}]: {conds_desc}"
        rule = {
            'type':        'temporal',
            'name':        name,
            'conditions':  pat['conditions'],
            'numbers':     pat['numbers'],
            'min_matches': 2,
        }
        cur.execute(
            'INSERT INTO rules(name,description,rule_json,created_at) VALUES(?,?,?,?)',
            (name, f"n_draws={pat.get('n_draws','?')}", json.dumps(rule), now),
        )
        new_ids.append(cur.lastrowid)
        logging.info('Temporal rule: %s → %s', conds_desc, pat['numbers'][:4])
    return new_ids


# ---------------------------------------------------------------------------
# Lag pattern discovery  (T-1 weight 3, T-2 weight 2, T-3 weight 1)
# ---------------------------------------------------------------------------

def _discover_lag_patterns(enriched, min_lift=1.5, min_draws=5, top_n=6):
    """Find transition and recency patterns driven by T-1 / T-2 / T-3 draws.

    Three pattern classes:

    **transition** (per lag)
      Given source number S appeared in T-k, target number T appears at
      > min_lift × baseline rate.  Emits one rule per (source, lag) with all
      high-lift targets grouped together.  T-1 is explored first (most weight).

    **recency cluster**
      Numbers whose weighted recency score  (3·in_T1 + 2·in_T2 + 1·in_T3) is
      positively correlated with appearing in the current draw.

    **cold cluster**
      Numbers absent from T-1, T-2 *and* T-3 that nevertheless appear at above-
      baseline rates (contrarian / overdue from recent perspective).
    """
    GLOBAL_RATE = 6.0 / 49
    LAG_KEYS    = [('_t1', 1), ('_t2', 2), ('_t3', 3)]   # (feature_key, lag_number)
    WEIGHTS     = {1: 3, 2: 2, 3: 1}                      # T-1 has highest weight
    patterns    = []

    # ── transition patterns ──────────────────────────────────────────────────
    for lag_key, lag_k in LAG_KEYS:
        for source in range(1, 50):
            # draws where source appeared in T-k
            conditioned = [
                nums for _, f, nums in enriched
                if source in f.get(lag_key, frozenset())
            ]
            if len(conditioned) < min_draws:
                continue
            n = len(conditioned)
            counts = Counter(t for nums in conditioned for t in nums)
            hot = sorted(
                [(t, counts[t] / n / GLOBAL_RATE)
                 for t in range(1, 50)
                 if counts[t] / n >= GLOBAL_RATE * min_lift],
                key=lambda x: -x[1],
            )
            if hot:
                patterns.append({
                    'lag_conditions': [{'lag': lag_k, 'numbers': [source], 'min_present': 1}],
                    'numbers':        [t for t, _ in hot[:top_n]],
                    'max_lift':       hot[0][1],
                    'n_draws':        n,
                    'source':         f'lag_transition_t{lag_k}',
                    'weight':         WEIGHTS[lag_k],
                })

    # ── recency cluster  (high weighted-recency → appears in current draw) ───
    recency_hits   = Counter()   # number → times recency > 0 AND appeared
    recency_trials = Counter()   # number → times recency > 0
    for _, f, nums in enriched:
        for n in range(1, 50):
            score = (WEIGHTS[1] * (n in f.get('_t1', frozenset())) +
                     WEIGHTS[2] * (n in f.get('_t2', frozenset())) +
                     WEIGHTS[3] * (n in f.get('_t3', frozenset())))
            if score > 0:
                recency_trials[n] += 1
                if n in nums:
                    recency_hits[n] += 1

    hot_recency = sorted(
        [(n, recency_hits[n] / recency_trials[n] / GLOBAL_RATE)
         for n in range(1, 50)
         if recency_trials[n] >= min_draws
         and recency_hits[n] / recency_trials[n] >= GLOBAL_RATE * min_lift],
        key=lambda x: -x[1],
    )
    if len(hot_recency) >= 2:
        hot_nums = [n for n, _ in hot_recency[:top_n]]
        patterns.append({
            'lag_conditions': [{'lag': 1, 'numbers': hot_nums, 'min_present': 1}],
            'numbers':        hot_nums,
            'max_lift':       hot_recency[0][1],
            'n_draws':        len(enriched),
            'source':         'lag_recency',
            'weight':         WEIGHTS[1],
        })

    # ── cold cluster  (absent T-1/T-2/T-3 → still appears) ──────────────────
    cold_hits   = Counter()
    cold_trials = Counter()
    for _, f, nums in enriched:
        t1 = f.get('_t1', frozenset())
        t2 = f.get('_t2', frozenset())
        t3 = f.get('_t3', frozenset())
        for n in range(1, 50):
            if n not in t1 and n not in t2 and n not in t3:
                cold_trials[n] += 1
                if n in nums:
                    cold_hits[n] += 1

    cold_numbers = sorted(
        [(n, cold_hits[n] / cold_trials[n] / GLOBAL_RATE)
         for n in range(1, 50)
         if cold_trials[n] >= min_draws
         and cold_hits[n] / cold_trials[n] >= GLOBAL_RATE * min_lift],
        key=lambda x: -x[1],
    )
    if len(cold_numbers) >= 2:
        cold_nums = [n for n, _ in cold_numbers[:top_n]]
        patterns.append({
            'lag_conditions': [{'lag': 1, 'numbers': cold_nums, 'min_present': 0}],
            'numbers':        cold_nums,
            'max_lift':       cold_numbers[0][1],
            'n_draws':        len(enriched),
            'source':         'lag_cold',
            'weight':         1,
        })

    # Sort by weight desc, then lift desc — T-1 patterns surface first
    patterns.sort(key=lambda p: (-p.get('weight', 1), -p.get('max_lift', 0)))
    return patterns


def generate_lag_rules(cur, min_lift=1.5, min_draws=5, top_n=6):
    """Discover lag patterns from T-1/T-2/T-3 and persist them as 'lag' rules.

    Returns list of newly inserted rule IDs.
    """
    enriched = _load_enriched_draws(cur)
    if len(enriched) < min_draws * 2:
        logging.warning('Not enough draws for lag analysis (%d)', len(enriched))
        return []

    patterns = _discover_lag_patterns(enriched, min_lift, min_draws, top_n)
    new_ids  = []
    now      = datetime.utcnow().isoformat()

    for pat in patterns:
        conds_desc = '; '.join(
            f"T-{c['lag']} has≥{c['min_present']} of {c['numbers']}"
            for c in pat['lag_conditions']
        )
        name = f"Lag [{pat['source']}]: {conds_desc}"
        rule = {
            'type':           'lag',
            'name':           name,
            'lag_conditions': pat['lag_conditions'],
            'numbers':        pat['numbers'],
            'min_matches':    2,
        }
        lift_str = f"{pat.get('max_lift', 0):.2f}"
        cur.execute(
            'INSERT INTO rules(name,description,rule_json,created_at) VALUES(?,?,?,?)',
            (name, f"n={pat.get('n_draws','?')} lift={lift_str}", json.dumps(rule), now),
        )
        new_ids.append(cur.lastrowid)
        logging.info('Lag rule [%s]: %s → %s', pat['source'], conds_desc, pat['numbers'][:4])
    return new_ids


def generate_correlation_rules(cur, min_lift=1.30, min_draws=15, top_n=8):
    """Discover pair-sum and pair-diff correlation patterns from all draw history.

    For each structural condition C (any pair sums to S; any pair differs by D):
      - Single variant:      condition fires in T-1 → check next-draw lift
      - Consecutive variant: condition fires in BOTH T-1 and T-2 → stronger signal

    Only rules with at least min_draws qualifying draws AND at least one target
    number with lift >= min_lift are kept.  Returns list of inserted rule IDs.
    """
    cur.execute("""
        SELECT d.draw_no, GROUP_CONCAT(j.number, ',')
        FROM draws d JOIN jackpot_no j ON d.draw_no=j.draw_no AND j.no_type='normal'
        WHERE d.scanned=1 GROUP BY d.draw_no ORDER BY d.draw_no
    """)
    rows      = cur.fetchall()
    all_draws = [(r[0], [int(n) for n in r[1].split(',')]) for r in rows]
    N         = len(all_draws)
    BASELINE  = 6 / 49

    if N < min_draws * 2:
        logging.warning('Not enough draws for correlation analysis (%d)', N)
        return []

    # Precompute structural signatures for every draw (set of pair sums, set of pair diffs)
    draw_sums  = [{a + b       for a, b in combinations(nums, 2)} for _, nums in all_draws]
    draw_diffs = [{abs(a - b)  for a, b in combinations(nums, 2)} for _, nums in all_draws]
    next_nums  = [all_draws[i + 1][1] if i + 1 < N else [] for i in range(N)]

    new_ids = []
    now     = datetime.utcnow().isoformat()

    def _scan(sig_sets, cond_type, val_range):
        """Yield rule dicts for each value V in val_range that has significant lift."""
        for V in val_range:
            # ── single-draw condition ────────────────────────────────────────
            counts, total = Counter(), 0
            for i in range(N - 1):
                if V in sig_sets[i]:
                    total += 1
                    for n in next_nums[i]:
                        counts[n] += 1
            if total >= min_draws:
                hits = sorted(
                    [(n, counts[n]/total, counts[n]/total/BASELINE)
                     for n in range(1, 50)
                     if counts[n]/total >= BASELINE * min_lift],
                    key=lambda x: -x[2]
                )
                if hits:
                    yield {
                        'cond_type':   cond_type,
                        'cond_value':  V,
                        'consecutive': False,
                        'numbers':     [h[0] for h in hits[:top_n]],
                        'lifts':       [round(h[2], 3) for h in hits[:top_n]],
                        'n_draws':     total,
                        'max_lift':    hits[0][2],
                    }

            # ── consecutive-draw condition (both T-1 and T-2 fire) ────────────
            c_counts, c_total = Counter(), 0
            for i in range(1, N - 1):
                if V in sig_sets[i] and V in sig_sets[i - 1]:
                    c_total += 1
                    for n in next_nums[i]:
                        c_counts[n] += 1
            if c_total >= min_draws:
                c_hits = sorted(
                    [(n, c_counts[n]/c_total, c_counts[n]/c_total/BASELINE)
                     for n in range(1, 50)
                     if c_counts[n]/c_total >= BASELINE * min_lift],
                    key=lambda x: -x[2]
                )
                if c_hits:
                    yield {
                        'cond_type':   cond_type,
                        'cond_value':  V,
                        'consecutive': True,
                        'numbers':     [h[0] for h in c_hits[:top_n]],
                        'lifts':       [round(h[2], 3) for h in c_hits[:top_n]],
                        'n_draws':     c_total,
                        'max_lift':    c_hits[0][2],
                    }

    patterns = list(_scan(draw_sums,  'pair_sum',  range(3,  98)))
    patterns += list(_scan(draw_diffs, 'pair_diff', range(1,  49)))

    for pat in patterns:
        consec  = pat['consecutive']
        ctype   = pat['cond_type']
        cval    = pat['cond_value']
        name    = f"Corr [{ctype}={cval}{'(consec)' if consec else ''}]: → {pat['numbers'][:4]}"
        rule    = {
            'type':        'correlation',
            'cond_type':   ctype,
            'cond_value':  cval,
            'consecutive': consec,
            'numbers':     pat['numbers'],
            'lifts':       pat['lifts'],
            'min_matches': 1,
        }
        cur.execute(
            'INSERT INTO rules(name,description,rule_json,created_at) VALUES(?,?,?,?)',
            (name, f"n={pat['n_draws']} max_lift={pat['max_lift']:.2f}", json.dumps(rule), now),
        )
        new_ids.append(cur.lastrowid)
        logging.info('Correlation rule: %s', name)

    logging.info('Generated %d correlation rules (%d sum, %d diff)',
                 len(new_ids),
                 sum(1 for p in patterns if p['cond_type'] == 'pair_sum'),
                 sum(1 for p in patterns if p['cond_type'] == 'pair_diff'))
    return new_ids


def generate_rules(cur, window=50, top_freq=10, top_pairs=5, min_gap=15):
    """Auto-generate rules from observed patterns in draw history.

    Generates five rule classes:
    - **frequency**:   top ``top_freq`` hot numbers in the last ``window`` draws.
    - **pair**:        top ``top_pairs`` most co-appearing pairs.
    - **gap**:         numbers overdue by >= ``min_gap`` draws.
    - **temporal**:    numbers correlated with draw date features.
    - **lag**:         numbers correlated with T-1/T-2/T-3 draw numbers.
    - **correlation**: numbers elevated after draws with specific pair sums/diffs.

    Returns the list of newly inserted rule IDs.
    """
    cur.execute('SELECT draw_no, COUNT(*) FROM draws WHERE scanned=1 GROUP BY draw_no ORDER BY draw_no DESC LIMIT ?', (window,))
    recent_draws = [r[0] for r in cur.fetchall()]

    if not recent_draws:
        logging.warning('No scanned draws found; cannot generate rules.')
        return []

    # collect numbers per draw
    placeholders = ','.join('?' * len(recent_draws))
    cur.execute(
        f'SELECT draw_no, number FROM jackpot_no WHERE no_type="normal" AND draw_no IN ({placeholders})',
        recent_draws,
    )
    draw_numbers = defaultdict(list)
    for draw_no, number in cur.fetchall():
        draw_numbers[draw_no].append(int(number))

    freq = Counter()
    pair_counts = Counter()
    for numbers in draw_numbers.values():
        for n in numbers:
            freq[n] += 1
        for a, b in combinations(sorted(numbers), 2):
            pair_counts[(a, b)] += 1

    new_ids = []
    now = datetime.utcnow().isoformat()

    # --- frequency rule: top N hot numbers ---
    hot = [n for n, _ in freq.most_common(top_freq)]
    if hot:
        rule = {
            'type': 'frequency',
            'name': f'Hot-{top_freq} (last {window} draws)',
            'numbers': hot,
            'min_matches': 3,
            'description': f'At least 3 of the {top_freq} most frequent numbers in the last {window} draws appear',
        }
        cur.execute(
            'INSERT INTO rules(name,description,rule_json,created_at) VALUES(?,?,?,?)',
            (rule['name'], rule['description'], json.dumps(rule), now),
        )
        new_ids.append(cur.lastrowid)
        logging.info('Generated frequency rule: %s', rule['name'])

    # --- pair rules: top M hot pairs ---
    for (a, b), count in pair_counts.most_common(top_pairs):
        rule = {
            'type': 'pair',
            'name': f'Hot pair ({a},{b}) — seen {count}x',
            'number_a': a,
            'number_b': b,
            'description': f'Numbers {a} and {b} co-appeared {count} times in the last {window} draws',
        }
        cur.execute(
            'INSERT INTO rules(name,description,rule_json,created_at) VALUES(?,?,?,?)',
            (rule['name'], rule['description'], json.dumps(rule), now),
        )
        new_ids.append(cur.lastrowid)
        logging.info('Generated pair rule: (%d, %d) count=%d', a, b, count)

    # --- gap rule: overdue numbers ---
    cur.execute('SELECT draw_no FROM draws WHERE scanned=1 ORDER BY draw_no DESC LIMIT 1')
    latest_row = cur.fetchone()
    if latest_row:
        latest = latest_row[0]
        cur.execute(
            """SELECT jn.number,
                      MAX(jn.draw_no) AS last_seen
               FROM jackpot_no jn
               WHERE jn.no_type='normal'
               GROUP BY jn.number""",
        )
        gap_numbers = [int(row[0]) for row in cur.fetchall() if (latest - row[1]) >= min_gap]
        if gap_numbers:
            rule = {
                'type': 'gap',
                'name': f'Overdue numbers (gap >= {min_gap})',
                'min_gap': min_gap,
                'description': f'At least one winning number had not appeared for {min_gap}+ draws before this draw',
            }
            cur.execute(
                'INSERT INTO rules(name,description,rule_json,created_at) VALUES(?,?,?,?)',
                (rule['name'], rule['description'], json.dumps(rule), now),
            )
            new_ids.append(cur.lastrowid)
            logging.info('Generated gap rule: %d overdue numbers', len(gap_numbers))

    # --- temporal rules: AI-based date-feature pattern discovery ---
    temporal_ids = generate_temporal_rules(cur)
    new_ids.extend(temporal_ids)

    # --- lag rules: T-1/T-2/T-3 transition and recency patterns ---
    lag_ids = generate_lag_rules(cur)
    new_ids.extend(lag_ids)

    # --- correlation rules: pair-sum and pair-diff structural patterns ---
    corr_ids = generate_correlation_rules(cur)
    new_ids.extend(corr_ids)

    # --- bucket distribution rules: per-bucket number counts ---
    bucket_ids = generate_bucket_rules(cur)
    new_ids.extend(bucket_ids)

    # --- odd/even count rules ---
    oe_ids = generate_odd_even_rules(cur)
    new_ids.extend(oe_ids)

    # --- seed-number derivation rules ---
    seed_ids = generate_seed_rules(cur)
    new_ids.extend(seed_ids)

    return new_ids


def generate_seed_rules(cur, min_lift=1.30, min_draws=15, top_n=8):
    """Discover seed-number patterns from draw history.

    A draw 'has seed S' when S (1-9) is one of the 6 winning numbers.
    For each (seed, method, min_derived) the rule fires when T-1 has seed S
    and at least min_derived of its numbers are derivable from S.

    Methods: 'multiple', 'last_digit', 'digit_sum'.
    min_derived range tested: 2 and 3 (seed itself counts as 1).

    Returns list of newly inserted rule IDs.
    """
    cur.execute("""
        SELECT d.draw_no, GROUP_CONCAT(j.number, ',')
        FROM draws d JOIN jackpot_no j ON d.draw_no=j.draw_no AND j.no_type='normal'
        WHERE d.scanned=1 GROUP BY d.draw_no ORDER BY d.draw_no
    """)
    rows      = cur.fetchall()
    all_draws = [(r[0], [int(n) for n in r[1].split(',')]) for r in rows]
    N         = len(all_draws)
    BASELINE  = 6 / 49
    if N < min_draws * 2:
        return []

    next_nums = [all_draws[i + 1][1] if i + 1 < N else [] for i in range(N)]
    new_ids   = []
    now       = datetime.utcnow().isoformat()

    active_seeds = load_seeds()
    logging.info('Generating seed rules for seeds: %s', active_seeds)

    for seed in active_seeds:
        for method in SEED_DERIVATIONS:
            derived_set = SEED_DERIVATIONS[method](seed)
            # Precompute coverage per draw
            coverages = [len(_seed_coverage(nums, seed, method))
                         if seed in nums else -1
                         for _, nums in all_draws]

            for min_derived in [2, 3, 4]:
                cnts, total = Counter(), 0
                for i in range(N - 1):
                    if coverages[i] >= min_derived:
                        total += 1
                        for n in next_nums[i]:
                            cnts[n] += 1
                if total < min_draws:
                    continue
                hits = sorted(
                    [(n, cnts[n] / total, cnts[n] / total / BASELINE)
                     for n in range(1, 50) if cnts[n] / total >= BASELINE * min_lift],
                    key=lambda x: -x[2]
                )
                if not hits:
                    continue
                name = (f'Seed [{seed}/{method}/cov≥{min_derived}]: '
                        f'→ {[h[0] for h in hits[:4]]}')
                rule = {
                    'type':        'seed',
                    'seed':        seed,
                    'method':      method,
                    'min_derived': min_derived,
                    'derived_set': sorted(derived_set & set(range(1, 50))),
                    'numbers':     [h[0] for h in hits[:top_n]],
                    'lifts':       [round(h[2], 3) for h in hits[:top_n]],
                    'min_matches': 1,
                }
                cur.execute(
                    'INSERT INTO rules(name,description,rule_json,created_at) VALUES(?,?,?,?)',
                    (name, f"n={total} max_lift={hits[0][2]:.2f}", json.dumps(rule), now),
                )
                new_ids.append(cur.lastrowid)
                logging.info('Seed rule: %s', name)

    logging.info('Generated %d seed rules', len(new_ids))
    return new_ids


def generate_bucket_rules(cur, min_lift=1.30, min_draws=15, top_n=8):
    """Discover per-bucket count patterns from draw history.

    For each bucket B (0-4) and count k (0-6): when the T-1 draw has exactly k
    numbers in bucket B, which numbers appear elevated in the next draw?

    Returns list of newly inserted rule IDs.
    """
    cur.execute("""
        SELECT d.draw_no, GROUP_CONCAT(j.number, ',')
        FROM draws d JOIN jackpot_no j ON d.draw_no=j.draw_no AND j.no_type='normal'
        WHERE d.scanned=1 GROUP BY d.draw_no ORDER BY d.draw_no
    """)
    rows      = cur.fetchall()
    all_draws = [(r[0], [int(n) for n in r[1].split(',')]) for r in rows]
    N         = len(all_draws)
    BASELINE  = 6 / 49
    if N < min_draws * 2:
        return []

    dists     = [_bucket_dist(nums) for _, nums in all_draws]
    next_nums = [all_draws[i + 1][1] if i + 1 < N else [] for i in range(N)]
    new_ids   = []
    now       = datetime.utcnow().isoformat()

    for bucket in range(5):
        lo, hi    = BUCKET_RANGES[bucket]
        lbl       = BUCKET_LABELS[bucket]
        for count in range(7):          # 0 … 6 numbers in this bucket
            cnts, total = Counter(), 0
            for i in range(N - 1):
                if dists[i][bucket] == count:
                    total += 1
                    for n in next_nums[i]:
                        cnts[n] += 1
            if total < min_draws:
                continue
            hits = sorted(
                [(n, cnts[n] / total, cnts[n] / total / BASELINE)
                 for n in range(1, 50) if cnts[n] / total >= BASELINE * min_lift],
                key=lambda x: -x[2]
            )
            if not hits:
                continue
            name = f'Bucket [{lbl}]=={count}: → {[h[0] for h in hits[:4]]}'
            rule = {
                'type':         'bucket_count',
                'bucket':       bucket,
                'count':        count,
                'bucket_range': [lo, hi],
                'numbers':      [h[0] for h in hits[:top_n]],
                'lifts':        [round(h[2], 3) for h in hits[:top_n]],
                'min_matches':  1,
            }
            cur.execute(
                'INSERT INTO rules(name,description,rule_json,created_at) VALUES(?,?,?,?)',
                (name, f"n={total} max_lift={hits[0][2]:.2f}", json.dumps(rule), now),
            )
            new_ids.append(cur.lastrowid)

    logging.info('Generated %d bucket-count rules', len(new_ids))
    return new_ids


def generate_odd_even_rules(cur, min_lift=1.30, min_draws=15, top_n=8):
    """Discover odd/even count patterns from draw history.

    For each possible odd count k (0-6): when the T-1 draw has exactly k odd
    numbers, which numbers appear elevated in the next draw?

    Returns list of newly inserted rule IDs.
    """
    cur.execute("""
        SELECT d.draw_no, GROUP_CONCAT(j.number, ',')
        FROM draws d JOIN jackpot_no j ON d.draw_no=j.draw_no AND j.no_type='normal'
        WHERE d.scanned=1 GROUP BY d.draw_no ORDER BY d.draw_no
    """)
    rows      = cur.fetchall()
    all_draws = [(r[0], [int(n) for n in r[1].split(',')]) for r in rows]
    N         = len(all_draws)
    BASELINE  = 6 / 49
    if N < min_draws * 2:
        return []

    odd_cnts  = [_odd_count(nums) for _, nums in all_draws]
    next_nums = [all_draws[i + 1][1] if i + 1 < N else [] for i in range(N)]
    new_ids   = []
    now       = datetime.utcnow().isoformat()

    for k in range(7):              # 0 … 6 odd numbers in T-1
        cnts, total = Counter(), 0
        for i in range(N - 1):
            if odd_cnts[i] == k:
                total += 1
                for n in next_nums[i]:
                    cnts[n] += 1
        if total < min_draws:
            continue
        hits = sorted(
            [(n, cnts[n] / total, cnts[n] / total / BASELINE)
             for n in range(1, 50) if cnts[n] / total >= BASELINE * min_lift],
            key=lambda x: -x[2]
        )
        if not hits:
            continue
        name = f'OddEven [odd=={k}, even=={6-k}]: → {[h[0] for h in hits[:4]]}'
        rule = {
            'type':        'odd_even',
            'odd_count':   k,
            'even_count':  6 - k,
            'numbers':     [h[0] for h in hits[:top_n]],
            'lifts':       [round(h[2], 3) for h in hits[:top_n]],
            'min_matches': 1,
        }
        cur.execute(
            'INSERT INTO rules(name,description,rule_json,created_at) VALUES(?,?,?,?)',
            (name, f"n={total} max_lift={hits[0][2]:.2f}", json.dumps(rule), now),
        )
        new_ids.append(cur.lastrowid)

    logging.info('Generated %d odd/even rules', len(new_ids))
    return new_ids


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_next_draw(cur, top_n=6):
    """Score every number 1–49 for the next draw using all rules in the DB.

    Weights applied:
      lag rules    — T-1 source ×3.0, T-2 ×2.0, T-3 ×1.0  (base weight ×2.0)
      temporal     — base ×1.5 when next-draw date conditions match
      frequency    — base ×1.0 (always contributes)
      pair / gap   — base ×0.5

    Returns ``(ranked, rule_details)`` where:
      ranked       — list of ``(number, score)`` sorted descending, length top_n
      rule_details — dict  number → list of contributing rule name strings
    """
    # ── T-1 / T-2 / T-3 number sets ────────────────────────────────────────
    cur.execute(
        'SELECT draw_no, day, date FROM draws WHERE scanned=1 ORDER BY draw_no DESC LIMIT 3'
    )
    latest = cur.fetchall()
    if not latest:
        logging.warning('No scanned draws; run totoscrape.py first.')
        return [], {}

    lag = {}
    for k, (draw_no, _, _) in enumerate(latest, 1):
        cur.execute(
            'SELECT number FROM jackpot_no WHERE draw_no=? AND no_type="normal"',
            (draw_no,),
        )
        lag[k] = frozenset(int(r[0]) for r in cur.fetchall())

    t1_draw_no, t1_day, t1_date = latest[0]
    logging.info('Predicting from T-1=draw %d  T-1 numbers: %s',
                 t1_draw_no, sorted(lag.get(1, [])))

    # ── predicted-draw date features ────────────────────────────────────────
    t1_feat = _parse_date_features(t1_date, t1_day)
    pred_features = {}
    if t1_feat.get('next_dom') and t1_feat.get('next_month'):
        pred_day      = 'Thu' if t1_day == 'Mon' else 'Mon'
        inv_month     = {v: k for k, v in MONTH_ABBR.items()}
        year          = t1_date.strip().split()[-1]
        pred_date_str = (
            f"{t1_feat['next_dom']:02d} "
            f"{inv_month.get(t1_feat['next_month'], 'Jan')} {year}"
        )
        pred_features = _parse_date_features(
            pred_date_str, pred_day, prev_date_str=t1_date
        )
        logging.info('Predicted draw date: %s (%s)', pred_date_str, pred_day)

    # ── precompute currently overdue numbers for gap rules ──────────────────
    cur.execute(
        """SELECT CAST(jn.number AS INTEGER), MAX(jn.draw_no)
           FROM jackpot_no jn WHERE jn.no_type='normal'
           GROUP BY jn.number"""
    )
    last_seen = {int(r[0]): r[1] for r in cur.fetchall()}

    # ── score numbers using every rule ────────────────────────────────────
    LAG_WEIGHTS = {1: 3.0, 2: 2.0, 3: 1.0}
    TYPE_BASE   = {'lag': 2.0, 'temporal': 1.5, 'frequency': 1.0,
                   'pair': 0.5, 'gap': 0.5, 'correlation': 1.2,
                   'bucket_count': 1.0, 'odd_even': 0.8, 'seed': 1.1}

    # ── precompute T-1 structural features ──────────────────────────────────
    t1_nums_list   = sorted(lag.get(1, []))
    t1_bucket_dist = _bucket_dist(t1_nums_list)
    t1_odd_cnt     = _odd_count(t1_nums_list)
    logging.info('T-1 bucket dist %s  odd=%d even=%d',
                 dict(zip(BUCKET_LABELS, t1_bucket_dist)),
                 t1_odd_cnt, 6 - t1_odd_cnt)

    scores       = defaultdict(float)
    rule_details = defaultdict(list)

    cur.execute('SELECT name, rule_json FROM rules')
    for name, rule_json in cur.fetchall():
        rule      = json.loads(rule_json)
        rule_type = rule.get('type')
        base_w    = TYPE_BASE.get(rule_type, 0.5)

        if rule_type == 'lag':
            ok      = True
            min_lag = 3
            for cond in rule.get('lag_conditions', []):
                k     = cond['lag']
                needed = set(cond.get('numbers', []))
                min_p  = cond.get('min_present', 1)
                if k not in lag or len(needed & lag[k]) < min_p:
                    ok = False; break
                min_lag = min(min_lag, k)
            if ok:
                w = base_w * LAG_WEIGHTS[min_lag]
                for n in rule.get('numbers', []):
                    scores[n]       += w
                    rule_details[n].append(f'{name} (+{w:.1f})')

        elif rule_type == 'temporal' and pred_features:
            ok = True
            for cond in rule.get('conditions', []):
                fv  = pred_features.get(cond['feature'])
                op  = cond.get('op', 'eq')
                val = cond['value']
                if fv is None:                       ok = False; break
                if   op == 'eq'  and fv != val:      ok = False; break
                elif op == 'in'  and fv not in val:  ok = False; break
                elif op == 'gte' and fv < val:       ok = False; break
                elif op == 'lte' and fv > val:       ok = False; break
            if ok:
                for n in rule.get('numbers', []):
                    scores[n]       += base_w
                    rule_details[n].append(f'{name} (+{base_w:.1f})')

        elif rule_type == 'frequency':
            for n in rule.get('numbers', []):
                scores[n]       += base_w
                rule_details[n].append(f'{name} (+{base_w:.1f})')

        elif rule_type == 'pair':
            for n in [rule.get('number_a'), rule.get('number_b')]:
                if n is not None:
                    scores[n]       += base_w
                    rule_details[n].append(f'{name} (+{base_w:.1f})')

        elif rule_type == 'gap':
            min_gap = rule.get('min_gap', 10)
            for n, ls in last_seen.items():
                if t1_draw_no - ls >= min_gap:
                    scores[n]       += base_w
                    rule_details[n].append(f'{name} gap={t1_draw_no - ls} (+{base_w:.1f})')

        elif rule_type == 'correlation':
            cond_type   = rule.get('cond_type')
            cond_value  = rule.get('cond_value')
            consecutive = rule.get('consecutive', False)

            def _corr_fires(nums, ct=cond_type, cv=cond_value):
                if ct == 'pair_sum':
                    return any(a + b == cv for a, b in combinations(nums, 2))
                if ct == 'pair_diff':
                    return any(abs(a - b) == cv for a, b in combinations(nums, 2))
                return False

            if not _corr_fires(lag.get(1, frozenset())):
                continue
            if consecutive and not _corr_fires(lag.get(2, frozenset())):
                continue

            w = base_w * (2.0 if consecutive else 1.0)
            for n in rule.get('numbers', []):
                scores[n]       += w
                rule_details[n].append(f'{name} (+{w:.1f})')

        elif rule_type == 'bucket_count':
            bucket = rule.get('bucket')
            count  = rule.get('count')
            if bucket is not None and t1_bucket_dist[bucket] == count:
                for n in rule.get('numbers', []):
                    scores[n]       += base_w
                    rule_details[n].append(f'{name} (+{base_w:.1f})')

        elif rule_type == 'odd_even':
            if t1_odd_cnt == rule.get('odd_count'):
                for n in rule.get('numbers', []):
                    scores[n]       += base_w
                    rule_details[n].append(f'{name} (+{base_w:.1f})')

        elif rule_type == 'seed':
            seed        = rule.get('seed')
            method      = rule.get('method')
            min_derived = rule.get('min_derived', 2)
            if (isinstance(seed, int) and 1 <= seed <= 49
                    and method in SEED_DERIVATIONS
                    and seed in t1_nums_list):
                covered = _seed_coverage(t1_nums_list, seed, method)
                if len(covered) >= min_derived:
                    for n in rule.get('numbers', []):
                        scores[n]       += base_w
                        rule_details[n].append(
                            f'{name} seed={seed}/{method}/cov={len(covered)} (+{base_w:.1f})')

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return (ranked[:top_n] if top_n else ranked), dict(rule_details)


def main():
    parser = argparse.ArgumentParser(description='TOTO rule engine')
    parser.add_argument('--add',        metavar='JSON', help='Add a rule (JSON string)')
    parser.add_argument('--list',       action='store_true', help='List all rules')
    parser.add_argument('--run',        metavar='ID',   help='Run rule by ID, or "all"')
    parser.add_argument('--generate',   action='store_true',
                        help='Auto-generate frequency/pair/gap rules from draw history')
    parser.add_argument('--window',     type=int, default=50,
                        help='Number of recent draws to analyse for --generate (default 50)')
    parser.add_argument('--predict',    action='store_true',
                        help='Score and rank numbers for the next draw using all rules')
    parser.add_argument('--top',        type=int, default=6,
                        help='How many candidates to show with --predict (default 6)')
    parser.add_argument('--favourites', metavar='N1,N2,...',
                        help='Set favourite numbers (comma-separated, 1-49) and save to favourites.json')
    parser.add_argument('--focus',      action='store_true',
                        help='With --predict: show only favourite numbers instead of full top-N')
    parser.add_argument('--seeds',      metavar='N1,N2,...',
                        help='Add extra seed numbers beyond the default 1-9 (saves to seeds.json)')
    args = parser.parse_args()

    # ── Handle --seeds (set extra seeds) ──────────────────────────────────
    if args.seeds:
        try:
            all_seeds = save_seeds([int(x.strip()) for x in args.seeds.split(',')])
            print(f'Seeds saved  (default 1-9 always included): {all_seeds}')
        except ValueError as e:
            print(f'Error: {e}')
            return
        if not args.generate and not args.predict:
            return

    # ── Handle --favourites (set / save) ────────────────────────────────────
    if args.favourites:
        try:
            nums = save_favourites([int(x.strip()) for x in args.favourites.split(',')])
            print(f'Favourites saved: {nums}')
        except ValueError as e:
            print(f'Error: {e}')
            return
        # Fall through to --predict if that flag is also given, otherwise stop
        if not args.predict:
            return

    conn = sqlite3.connect('toto.sqlite')
    try:
        cur = conn.cursor()

        if args.add:
            rule_id = add_rule(cur, args.add)
            conn.commit()
            logging.info('Rule added with ID %d', rule_id)

        elif args.list:
            rows = list_rules(cur)
            if not rows:
                print('No rules defined.')
            for rule_id, name, desc, created_at in rows:
                print(f'[{rule_id}] {name}  ({created_at})')
                if desc:
                    print(f'    {desc}')

        elif args.generate:
            new_ids = generate_rules(cur, window=args.window)
            conn.commit()
            logging.info('Generated %d new rules', len(new_ids))

        elif args.predict:
            favourites = load_favourites()

            if args.focus and favourites:
                # Focus mode: score all but display only favourites
                ranked, rule_details = predict_next_draw(cur, top_n=0)
                if not ranked:
                    print('No rules or no scanned draws. Run --generate first.')
                else:
                    t1_row = cur.execute(
                        'SELECT draw_no, date FROM draws WHERE scanned=1 ORDER BY draw_no DESC LIMIT 1'
                    ).fetchone()
                    print(f'\nPrediction based on T-1 = draw {t1_row[0]} ({t1_row[1]})')
                    print_favourite_analysis(cur, favourites, ranked, rule_details)
            else:
                # Normal mode: show top-N + favourites section
                ranked, rule_details = predict_next_draw(cur, top_n=args.top)
                if not ranked:
                    print('No rules or no scanned draws. Run --generate first.')
                else:
                    t1_row = cur.execute(
                        'SELECT draw_no, date FROM draws WHERE scanned=1 ORDER BY draw_no DESC LIMIT 1'
                    ).fetchone()
                    print(f'\nPrediction based on T-1 = draw {t1_row[0]} ({t1_row[1]})')
                    print(f'{"#":<4} {"Num":<6} {"Score":<8}  Top contributing rules')
                    print('-' * 70)
                    for i, (number, score) in enumerate(ranked, 1):
                        top_rules = '; '.join(rule_details.get(number, [])[:2])
                        print(f'{i:<4} {number:<6} {score:<8.2f}  {top_rules}')
                    print(f'\nSuggested numbers: {[n for n, _ in ranked]}')

                    if favourites:
                        # Always show favourite analysis when configured
                        ranked_all, rule_details_all = predict_next_draw(cur, top_n=0)
                        print_favourite_analysis(cur, favourites, ranked_all, rule_details_all)

        elif args.run:
            if args.run == 'all':
                cur.execute('SELECT rule_id, rule_json FROM rules')
                rules = cur.fetchall()
            else:
                cur.execute('SELECT rule_id, rule_json FROM rules WHERE rule_id=?', (int(args.run),))
                rules = cur.fetchall()
            for rule_id, rule_json in rules:
                matched = evaluate_rule(cur, rule_id, rule_json)
                conn.commit()
                logging.info('Rule %d: %d draws matched', rule_id, matched)

        else:
            fav = load_favourites()
            if fav:
                print(f'Current favourites: {fav}')
                print('Use --predict [--focus] to see analysis.')
            parser.print_help()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _make_db(draws_and_numbers):
    """Create an in-memory DB with draws and jackpot_no rows for testing."""
    conn = sqlite3.connect(':memory:')
    cur = conn.cursor()
    cur.executescript('''
        CREATE TABLE draws(draw_no INTEGER PRIMARY KEY, day TEXT, date TEXT, scanned INTEGER DEFAULT 0);
        CREATE TABLE jackpot_no(draw_no INTEGER, no_type TEXT, number INTEGER);
        CREATE TABLE rules(rule_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT,
                           description TEXT, rule_json TEXT, created_at TEXT);
        CREATE TABLE rule_results(rule_id INTEGER, draw_no INTEGER, matched INTEGER,
                                  details_json TEXT, PRIMARY KEY(rule_id,draw_no));
    ''')
    for draw_no, numbers in draws_and_numbers.items():
        cur.execute('INSERT INTO draws VALUES(?,?,?,1)', (draw_no, 'Mon', f'01 Jan {draw_no}'))
        for n in numbers:
            cur.execute('INSERT INTO jackpot_no VALUES(?,?,?)', (draw_no, 'normal', n))
    return conn, cur


class TestEvalFrequencyRule(unittest.TestCase):
    def setUp(self):
        self.conn, self.cur = _make_db({
            1: [1, 2, 3, 4, 5, 6],
            2: [1, 7, 8, 9, 10, 11],
        })

    def test_match_when_enough_numbers_present(self):
        rule = {'type': 'frequency', 'numbers': [1, 2], 'min_matches': 2}
        matched, details = eval_frequency_rule(self.cur, 1, rule)
        self.assertTrue(matched)
        self.assertEqual(sorted(details['matched_numbers']), [1, 2])

    def test_no_match_when_below_threshold(self):
        rule = {'type': 'frequency', 'numbers': [1, 2], 'min_matches': 2}
        matched, _ = eval_frequency_rule(self.cur, 2, rule)
        self.assertFalse(matched)  # only 1 is present in draw 2

    def test_partial_match(self):
        rule = {'type': 'frequency', 'numbers': [1, 2], 'min_matches': 1}
        matched, details = eval_frequency_rule(self.cur, 2, rule)
        self.assertTrue(matched)
        self.assertEqual(details['matched_numbers'], [1])


class TestEvalPairRule(unittest.TestCase):
    def setUp(self):
        self.conn, self.cur = _make_db({
            1: [7, 14, 21, 28, 35, 42],
            2: [7,  3, 21, 28, 35, 42],
        })

    def test_pair_present(self):
        rule = {'type': 'pair', 'number_a': 7, 'number_b': 14}
        matched, _ = eval_pair_rule(self.cur, 1, rule)
        self.assertTrue(matched)

    def test_pair_absent(self):
        rule = {'type': 'pair', 'number_a': 7, 'number_b': 14}
        matched, _ = eval_pair_rule(self.cur, 2, rule)
        self.assertFalse(matched)


class TestEvalGapRule(unittest.TestCase):
    def setUp(self):
        self.conn, self.cur = _make_db({
            1:  [7, 1, 2, 3, 4, 5],
            20: [7, 8, 9, 10, 11, 12],  # gap of 19 for number 7
        })

    def test_gap_detected(self):
        rule = {'type': 'gap', 'min_gap': 10}
        matched, details = eval_gap_rule(self.cur, 20, rule)
        self.assertTrue(matched)
        self.assertEqual(details['overdue_numbers'][0]['number'], 7)
        self.assertEqual(details['overdue_numbers'][0]['gap'], 19)

    def test_gap_below_threshold_not_matched(self):
        rule = {'type': 'gap', 'min_gap': 30}
        matched, _ = eval_gap_rule(self.cur, 20, rule)
        self.assertFalse(matched)


class TestEvaluateRule(unittest.TestCase):
    def test_counts_matched_draws(self):
        conn, cur = _make_db({
            1: [7, 14, 1, 2, 3, 4],
            2: [7, 14, 5, 6, 8, 9],
            3: [1,  2, 3, 4, 5, 6],
        })
        rule_json = json.dumps({'type': 'pair', 'number_a': 7, 'number_b': 14})
        cur.execute('INSERT INTO rules VALUES(1,"test","",?,?)', (rule_json, '2024-01-01'))
        matched = evaluate_rule(cur, 1, rule_json)
        self.assertEqual(matched, 2)


class TestParseDateFeatures(unittest.TestCase):
    def test_basic_dom_and_month(self):
        f = _parse_date_features('08 Apr 2024')
        self.assertEqual(f['dom'], 8)
        self.assertEqual(f['month'], 4)

    def test_prev_date_adds_features(self):
        f = _parse_date_features('08 Apr 2024', prev_date_str='04 Apr 2024')
        self.assertEqual(f['prev_dom'], 4)
        self.assertEqual(f['prev_month'], 4)

    def test_next_computed_from_monday(self):
        # Mon 08 Apr 2024 → next TOTO draw is Thu 11 Apr 2024
        f = _parse_date_features('08 Apr 2024', day_str='Mon')
        self.assertEqual(f['next_dom'], 11)
        self.assertEqual(f['next_month'], 4)

    def test_next_computed_from_thursday(self):
        # Thu 11 Apr 2024 → next TOTO draw is Mon 15 Apr 2024
        f = _parse_date_features('11 Apr 2024', day_str='Thu')
        self.assertEqual(f['next_dom'], 15)
        self.assertEqual(f['next_month'], 4)

    def test_next_date_str_overrides_computed(self):
        f = _parse_date_features('08 Apr 2024', day_str='Mon', next_date_str='12 Apr 2024')
        self.assertEqual(f['next_dom'], 12)

    def test_month_boundary_next_draw(self):
        # Thu 30 Jan 2025 → next Mon is 03 Feb 2025
        f = _parse_date_features('30 Jan 2025', day_str='Thu')
        self.assertEqual(f['next_dom'], 3)
        self.assertEqual(f['next_month'], 2)


class TestEvalTemporalRule(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        cur = self.conn.cursor()
        cur.executescript('''
            CREATE TABLE draws(draw_no INTEGER PRIMARY KEY, day TEXT, date TEXT, scanned INTEGER DEFAULT 0);
            CREATE TABLE jackpot_no(draw_no INTEGER, no_type TEXT, number INTEGER);
        ''')
        # draw 100: Mon 01 Apr 2024  — month=4, dom=1
        cur.execute('INSERT INTO draws VALUES(100,"Mon","01 Apr 2024",1)')
        for n in [7, 14, 21, 28, 35, 42]:
            cur.execute('INSERT INTO jackpot_no VALUES(100,"normal",?)', (n,))
        # draw 101: Thu 04 Apr 2024  — prev_dom=1, prev_month=4
        cur.execute('INSERT INTO draws VALUES(101,"Thu","04 Apr 2024",1)')
        for n in [1, 2, 3, 4, 5, 6]:
            cur.execute('INSERT INTO jackpot_no VALUES(101,"normal",?)', (n,))
        self.cur = cur

    def test_matches_when_conditions_and_numbers_met(self):
        rule = {'type': 'temporal',
                'conditions': [{'feature': 'month', 'op': 'eq', 'value': 4}],
                'numbers': [7, 14], 'min_matches': 2}
        matched, details = eval_temporal_rule(self.cur, 100, rule)
        self.assertTrue(matched)
        self.assertIn(7, details['matched_numbers'])

    def test_no_match_when_condition_fails(self):
        rule = {'type': 'temporal',
                'conditions': [{'feature': 'month', 'op': 'eq', 'value': 5}],
                'numbers': [7, 14], 'min_matches': 1}
        matched, _ = eval_temporal_rule(self.cur, 100, rule)
        self.assertFalse(matched)

    def test_no_match_when_numbers_absent(self):
        rule = {'type': 'temporal',
                'conditions': [{'feature': 'month', 'op': 'eq', 'value': 4}],
                'numbers': [99, 98], 'min_matches': 1}
        matched, _ = eval_temporal_rule(self.cur, 100, rule)
        self.assertFalse(matched)

    def test_gte_condition(self):
        rule = {'type': 'temporal',
                'conditions': [{'feature': 'dom', 'op': 'gte', 'value': 1}],
                'numbers': [1, 2, 3], 'min_matches': 3}
        matched, _ = eval_temporal_rule(self.cur, 101, rule)
        self.assertTrue(matched)

    def test_prev_dom_feature_populated(self):
        # draw 101 follows draw 100 (dom=1); prev_dom should be 1
        rule = {'type': 'temporal',
                'conditions': [{'feature': 'prev_dom', 'op': 'eq', 'value': 1}],
                'numbers': [1], 'min_matches': 1}
        matched, _ = eval_temporal_rule(self.cur, 101, rule)
        self.assertTrue(matched)


class TestDiscoverTemporalPatterns(unittest.TestCase):
    def setUp(self):
        """10 draws in April — number 7 appears in 8 of them.
        5 draws in July — number 7 never appears.
        Baseline rate for number 7 ≈ 8/15 ≈ 53% vs global 12.2% → lift ≈ 4.4.
        """
        self.conn = sqlite3.connect(':memory:')
        cur = self.conn.cursor()
        cur.executescript('''
            CREATE TABLE draws(draw_no INTEGER PRIMARY KEY, day TEXT, date TEXT, scanned INTEGER DEFAULT 0);
            CREATE TABLE jackpot_no(draw_no INTEGER, no_type TEXT, number INTEGER);
        ''')
        for i in range(10):
            cur.execute('INSERT INTO draws VALUES(?,?,?,1)',
                        (i + 1, 'Mon', f'{i+1:02d} Apr 2024'))
            nums = [7, 14, 21, 28, 35, 42] if i < 8 else [1, 2, 3, 4, 5, 6]
            for n in nums:
                cur.execute('INSERT INTO jackpot_no VALUES(?,"normal",?)', (i + 1, n))
        for i in range(5):
            cur.execute('INSERT INTO draws VALUES(?,?,?,1)',
                        (i + 100, 'Mon', f'{i+1:02d} Jul 2024'))
            for n in [1, 2, 3, 4, 5, 6]:
                cur.execute('INSERT INTO jackpot_no VALUES(?,"normal",?)', (i + 100, n))
        self.cur = cur

    def test_discovers_hot_number_in_april(self):
        enriched = _load_enriched_draws(self.cur)
        patterns = _discover_by_frequency(enriched, min_lift=1.5, min_draws=5, top_n=6)
        april_pats = [p for p in patterns
                      if p['conditions'] == [{'feature': 'month', 'op': 'eq', 'value': 4}]]
        self.assertTrue(len(april_pats) > 0, 'Expected a pattern for April')
        self.assertIn(7, april_pats[0]['numbers'])

    def test_no_patterns_when_insufficient_draws(self):
        enriched = _load_enriched_draws(self.cur)
        patterns = _discover_by_frequency(enriched, min_lift=1.5, min_draws=20, top_n=6)
        self.assertEqual(patterns, [])

    def test_empty_when_no_scanned_draws(self):
        conn = sqlite3.connect(':memory:')
        cur = conn.cursor()
        cur.executescript('''
            CREATE TABLE draws(draw_no INTEGER PRIMARY KEY, day TEXT, date TEXT, scanned INTEGER DEFAULT 0);
            CREATE TABLE jackpot_no(draw_no INTEGER, no_type TEXT, number INTEGER);
        ''')
        enriched = _load_enriched_draws(cur)
        self.assertEqual(enriched, [])


class TestEvalLagRule(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        cur = self.conn.cursor()
        cur.executescript('''
            CREATE TABLE draws(draw_no INTEGER PRIMARY KEY, day TEXT, date TEXT, scanned INTEGER DEFAULT 0);
            CREATE TABLE jackpot_no(draw_no INTEGER, no_type TEXT, number INTEGER);
        ''')
        # T-3=draw1, T-2=draw2, T-1=draw3, current=draw4
        data = {
            1: [7, 14, 21, 28, 35, 42],
            2: [1,  2,  3,  4,  5,  6],
            3: [7,  8,  9, 10, 11, 12],   # 7 appears in T-1
            4: [7, 13, 14, 15, 16, 17],   # current draw — 7 and 14 appear
        }
        for draw_no, nums in data.items():
            cur.execute('INSERT INTO draws VALUES(?,"Mon","01 Jan 2024",1)', (draw_no,))
            for n in nums:
                cur.execute('INSERT INTO jackpot_no VALUES(?,"normal",?)', (draw_no, n))
        self.cur = cur

    def test_matches_when_t1_condition_met(self):
        rule = {'type': 'lag',
                'lag_conditions': [{'lag': 1, 'numbers': [7], 'min_present': 1}],
                'numbers': [7, 14], 'min_matches': 2}
        matched, details = eval_lag_rule(self.cur, 4, rule)
        self.assertTrue(matched)
        self.assertIn(7, details['matched_numbers'])

    def test_no_match_when_t1_source_absent(self):
        # number 99 was never in T-1
        rule = {'type': 'lag',
                'lag_conditions': [{'lag': 1, 'numbers': [99], 'min_present': 1}],
                'numbers': [7], 'min_matches': 1}
        matched, _ = eval_lag_rule(self.cur, 4, rule)
        self.assertFalse(matched)

    def test_t2_condition(self):
        # number 1 was in draw2 (T-2 relative to draw4)
        rule = {'type': 'lag',
                'lag_conditions': [{'lag': 2, 'numbers': [1], 'min_present': 1}],
                'numbers': [7], 'min_matches': 1}
        matched, _ = eval_lag_rule(self.cur, 4, rule)
        self.assertTrue(matched)

    def test_t3_condition(self):
        # number 42 was in draw1 (T-3 relative to draw4)
        rule = {'type': 'lag',
                'lag_conditions': [{'lag': 3, 'numbers': [42], 'min_present': 1}],
                'numbers': [7], 'min_matches': 1}
        matched, _ = eval_lag_rule(self.cur, 4, rule)
        self.assertTrue(matched)

    def test_absence_condition(self):
        # min_present=0 means source must NOT be present
        rule = {'type': 'lag',
                'lag_conditions': [{'lag': 1, 'numbers': [99], 'min_present': 0}],
                'numbers': [7], 'min_matches': 1}
        matched, _ = eval_lag_rule(self.cur, 4, rule)
        self.assertTrue(matched)   # 99 absent from T-1 → condition passes

    def test_lag_info_returned(self):
        rule = {'type': 'lag', 'lag_conditions': [], 'numbers': [7], 'min_matches': 1}
        _, details = eval_lag_rule(self.cur, 4, rule)
        self.assertIn(1, details['lag'])   # T-1 numbers present in details
        self.assertIn(7, details['lag'][1])


class TestDiscoverLagPatterns(unittest.TestCase):
    def _build_transition_db(self):
        """20 draws where number 7 in T-1 predicts number 14 at ~70% rate."""
        conn = sqlite3.connect(':memory:')
        cur  = conn.cursor()
        cur.executescript('''
            CREATE TABLE draws(draw_no INTEGER PRIMARY KEY, day TEXT, date TEXT, scanned INTEGER DEFAULT 0);
            CREATE TABLE jackpot_no(draw_no INTEGER, no_type TEXT, number INTEGER);
        ''')
        for i in range(30):
            cur.execute('INSERT INTO draws VALUES(?,"Mon",?,1)',
                        (i + 1, f'{(i%28)+1:02d} Jan 2024'))
            # 7 appears in even draws; 14 appears in the draw AFTER 7 appeared
            if i % 2 == 0:
                nums = [7, 21, 22, 23, 24, 25]
            else:
                nums = [14, 21, 22, 23, 24, 25]  # 14 follows 7
            for n in nums:
                cur.execute('INSERT INTO jackpot_no VALUES(?,"normal",?)', (i + 1, n))
        return conn, cur

    def test_transition_pattern_found(self):
        _, cur = self._build_transition_db()
        enriched = _load_enriched_draws(cur)
        patterns = _discover_lag_patterns(enriched, min_lift=1.5, min_draws=5)
        transition_pats = [p for p in patterns if 'lag_transition' in p['source']]
        self.assertTrue(len(transition_pats) > 0, 'Expected at least one transition pattern')
        # At least one rule should predict 14 when 7 was in T-1
        t1_pats = [p for p in transition_pats
                   if p['lag_conditions'][0]['lag'] == 1
                   and 7 in p['lag_conditions'][0]['numbers']
                   and 14 in p['numbers']]
        self.assertTrue(len(t1_pats) > 0, 'Expected 7→14 transition rule')

    def test_t1_rules_surface_before_t3(self):
        """T-1 patterns (weight 3) should appear before T-3 patterns (weight 1)."""
        _, cur = self._build_transition_db()
        enriched = _load_enriched_draws(cur)
        patterns = _discover_lag_patterns(enriched, min_lift=1.5, min_draws=5)
        transition_pats = [p for p in patterns if 'lag_transition' in p['source']]
        if len(transition_pats) >= 2:
            lags = [p['lag_conditions'][0]['lag'] for p in transition_pats]
            # First encountered lag should be ≤ any later lag
            self.assertLessEqual(lags[0], lags[-1])


class TestPredictNextDraw(unittest.TestCase):
    def _make_predict_db(self):
        """DB with 4 draws and two rules that predict specific numbers."""
        conn = sqlite3.connect(':memory:')
        cur  = conn.cursor()
        cur.executescript('''
            CREATE TABLE draws(draw_no INTEGER PRIMARY KEY, day TEXT, date TEXT, scanned INTEGER DEFAULT 0);
            CREATE TABLE jackpot_no(draw_no INTEGER, no_type TEXT, number INTEGER);
            CREATE TABLE rules(rule_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT,
                               description TEXT, rule_json TEXT, created_at TEXT);
        ''')
        draws = [
            (1, 'Mon', '01 Jan 2024', [21, 22, 23, 24, 25, 26]),
            (2, 'Thu', '04 Jan 2024', [31, 32, 33, 34, 35, 36]),
            (3, 'Mon', '08 Jan 2024', [7,  8,  9, 10, 11, 12]),  # T-1
            (4, 'Thu', '11 Jan 2024', [7, 14, 21, 28, 35, 42]),  # T-0 (latest scanned)
        ]
        for draw_no, day, date, nums in draws:
            cur.execute('INSERT INTO draws VALUES(?,?,?,1)', (draw_no, day, date))
            for n in nums:
                cur.execute('INSERT INTO jackpot_no VALUES(?,"normal",?)', (draw_no, n))
        # Lag rule: if 7 in T-1 → predict [7, 14, 21]  (T-1 weight ×3 → score = 2×3 = 6.0)
        lag_rule = json.dumps({'type': 'lag',
                               'lag_conditions': [{'lag': 1, 'numbers': [7], 'min_present': 1}],
                               'numbers': [7, 14, 21], 'min_matches': 2})
        cur.execute('INSERT INTO rules VALUES(1,"lag-test","",?,"2024")', (lag_rule,))
        # Frequency rule: always predict [7, 49]  (base weight 1.0)
        freq_rule = json.dumps({'type': 'frequency', 'numbers': [7, 49], 'min_matches': 1})
        cur.execute('INSERT INTO rules VALUES(2,"freq-test","",?,"2024")', (freq_rule,))
        return conn, cur

    def test_lag_rule_scores_predicted_numbers(self):
        _, cur = self._make_predict_db()
        # T-1 is draw 4; its T-1 (draw 3) had 7 → lag rule fires
        ranked, details = predict_next_draw(cur, top_n=49)
        score_map = dict(ranked)
        # 7 should have lag score (6.0) + freq score (1.0) = 7.0
        self.assertGreater(score_map.get(7, 0), score_map.get(2, 0))
        self.assertIn(7, [n for n, _ in ranked[:3]])

    def test_t1_weight_exceeds_t3(self):
        """A lag rule firing on T-1 should outscore one firing on T-3."""
        conn = sqlite3.connect(':memory:')
        cur  = conn.cursor()
        cur.executescript('''
            CREATE TABLE draws(draw_no INTEGER PRIMARY KEY, day TEXT, date TEXT, scanned INTEGER DEFAULT 0);
            CREATE TABLE jackpot_no(draw_no INTEGER, no_type TEXT, number INTEGER);
            CREATE TABLE rules(rule_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT,
                               description TEXT, rule_json TEXT, created_at TEXT);
        ''')
        for draw_no, nums in [(1, [5, 6, 7, 8, 9, 10]),
                              (2, [11,12,13,14,15,16]),
                              (3, [17,18,19,20,21,22]),
                              (4, [23,24,25,26,27,28])]:
            cur.execute('INSERT INTO draws VALUES(?,"Mon","01 Jan 2024",1)', (draw_no,))
            for n in nums:
                cur.execute('INSERT INTO jackpot_no VALUES(?,"normal",?)', (draw_no, n))
        # Rule A: if 23 (in T-1) → predict 40  (T-1 weight: 2.0 × 3.0 = 6.0)
        r_t1 = json.dumps({'type': 'lag',
                           'lag_conditions': [{'lag': 1, 'numbers': [23], 'min_present': 1}],
                           'numbers': [40]})
        cur.execute('INSERT INTO rules VALUES(1,"t1-rule","",?,"2024")', (r_t1,))
        # Rule B: if 5 (in T-3) → predict 41  (T-3 weight: 2.0 × 1.0 = 2.0)
        r_t3 = json.dumps({'type': 'lag',
                           'lag_conditions': [{'lag': 3, 'numbers': [5], 'min_present': 1}],
                           'numbers': [41]})
        cur.execute('INSERT INTO rules VALUES(2,"t3-rule","",?,"2024")', (r_t3,))
        ranked, _ = predict_next_draw(cur, top_n=49)
        score_map = dict(ranked)
        self.assertGreater(score_map.get(40, 0), score_map.get(41, 0))

    def test_no_draws_returns_empty(self):
        conn = sqlite3.connect(':memory:')
        cur  = conn.cursor()
        cur.executescript('''
            CREATE TABLE draws(draw_no INTEGER PRIMARY KEY, day TEXT, date TEXT, scanned INTEGER DEFAULT 0);
            CREATE TABLE jackpot_no(draw_no INTEGER, no_type TEXT, number INTEGER);
            CREATE TABLE rules(rule_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT,
                               description TEXT, rule_json TEXT, created_at TEXT);
        ''')
        ranked, details = predict_next_draw(cur, top_n=6)
        self.assertEqual(ranked, [])
        self.assertEqual(details, {})


class TestHelpers(unittest.TestCase):
    def test_bucket_dist_all_buckets(self):
        # One number per bucket
        self.assertEqual(_bucket_dist([1, 10, 20, 30, 40]), (1, 1, 1, 1, 1))

    def test_bucket_dist_boundary_values(self):
        # Boundary numbers land in the correct bucket
        self.assertEqual(_bucket_dist([9, 19, 29, 39, 49, 1]), (2, 1, 1, 1, 1))

    def test_bucket_dist_empty(self):
        self.assertEqual(_bucket_dist([]), (0, 0, 0, 0, 0))

    def test_odd_count(self):
        self.assertEqual(_odd_count([1, 2, 3, 4, 5, 6]), 3)

    def test_odd_count_all_odd(self):
        self.assertEqual(_odd_count([1, 3, 5, 7, 9, 11]), 6)

    def test_odd_count_all_even(self):
        self.assertEqual(_odd_count([2, 4, 6, 8, 10, 12]), 0)

    def test_seed_coverage_multiple(self):
        # seed=7: multiples in 1-49 are 7,14,21,28,35,42,49
        cov = _seed_coverage([7, 14, 21, 3, 5, 6], 7, 'multiple')
        self.assertEqual(cov, frozenset([7, 14, 21]))

    def test_seed_coverage_last_digit(self):
        # seed=3: numbers ending in 3 → 3,13,23,33,43
        cov = _seed_coverage([3, 13, 33, 5, 10], 3, 'last_digit')
        self.assertEqual(cov, frozenset([3, 13, 33]))

    def test_seed_coverage_digit_sum_single_digit(self):
        # seed=2: digit_sum=2 → 2,11,20,29,38,47
        cov = _seed_coverage([2, 11, 20, 5, 7], 2, 'digit_sum')
        self.assertEqual(cov, frozenset([2, 11, 20]))

    def test_seed_coverage_digit_sum_multi_digit(self):
        # seed=13: digit_sum=4 → 4,13,22,31,40,49
        cov = _seed_coverage([13, 22, 40, 7, 11], 13, 'digit_sum')
        self.assertEqual(cov, frozenset([13, 22, 40]))

    def test_seed_coverage_empty_nums(self):
        self.assertEqual(_seed_coverage([], 7, 'multiple'), frozenset())


class TestEvalCorrelationRule(unittest.TestCase):
    def setUp(self):
        # draw1=T-2, draw2=T-1, draw3=current
        # T-1 (draw2): [1,29,...] → pair_sum=30 fires
        # T-2 (draw1): [4,26,...] → pair_sum=30 fires
        self.conn, self.cur = _make_db({
            1: [4, 26, 3, 5, 6, 7],    # T-2: 4+26=30
            2: [1, 29, 8, 9, 10, 11],  # T-1: 1+29=30
            3: [15, 22, 33, 34, 35, 36],  # current
        })

    def test_pair_sum_fires_when_t1_matches(self):
        rule = {'type': 'correlation', 'cond_type': 'pair_sum', 'cond_value': 30,
                'consecutive': False, 'numbers': [15], 'min_matches': 1}
        matched, details = eval_correlation_rule(self.cur, 3, rule)
        self.assertTrue(matched)
        self.assertIn(15, details['matched_numbers'])

    def test_pair_sum_no_fire_when_t1_misses(self):
        # T-1 for draw3 = draw2 = [1,29,...] → pair_sum=30 is present
        # Let's try with a cond_value that is NOT present in T-1
        rule = {'type': 'correlation', 'cond_type': 'pair_sum', 'cond_value': 99,
                'consecutive': False, 'numbers': [15], 'min_matches': 1}
        matched, _ = eval_correlation_rule(self.cur, 3, rule)
        self.assertFalse(matched)

    def test_pair_diff_fires(self):
        # T-1 (draw2): |29-1|=28; use diff=28
        rule = {'type': 'correlation', 'cond_type': 'pair_diff', 'cond_value': 28,
                'consecutive': False, 'numbers': [15], 'min_matches': 1}
        matched, _ = eval_correlation_rule(self.cur, 3, rule)
        self.assertTrue(matched)

    def test_consecutive_fires_when_both_match(self):
        rule = {'type': 'correlation', 'cond_type': 'pair_sum', 'cond_value': 30,
                'consecutive': True, 'numbers': [15], 'min_matches': 1}
        matched, _ = eval_correlation_rule(self.cur, 3, rule)
        self.assertTrue(matched)   # both draw1 and draw2 have pair_sum=30

    def test_consecutive_fails_when_t2_misses(self):
        # draw1 (T-2) has 4+26=30, but we look for sum=99 which T-2 doesn't have
        conn, cur = _make_db({
            1: [3, 5, 7, 9, 11, 13],   # T-2: no pair sums to 30
            2: [1, 29, 8, 10, 12, 14], # T-1: 1+29=30
            3: [15, 22, 33, 34, 35, 36],
        })
        rule = {'type': 'correlation', 'cond_type': 'pair_sum', 'cond_value': 30,
                'consecutive': True, 'numbers': [15], 'min_matches': 1}
        matched, _ = eval_correlation_rule(cur, 3, rule)
        self.assertFalse(matched)

    def test_no_previous_draws_returns_false(self):
        conn, cur = _make_db({1: [1, 2, 3, 4, 5, 6]})
        rule = {'type': 'correlation', 'cond_type': 'pair_sum', 'cond_value': 3,
                'consecutive': False, 'numbers': [7], 'min_matches': 1}
        matched, _ = eval_correlation_rule(cur, 1, rule)
        self.assertFalse(matched)


class TestEvalBucketCountRule(unittest.TestCase):
    def setUp(self):
        # T-1 (draw1): [1,2,3,4,5,6] → bucket 0 (1-9) count=6
        # current (draw2): [10,11,12,13,14,15]
        self.conn, self.cur = _make_db({
            1: [1, 2, 3, 4, 5, 6],       # T-1: 6 numbers in bucket 0
            2: [10, 11, 12, 13, 14, 15],  # current
        })

    def test_fires_when_bucket_count_matches_exactly(self):
        rule = {'type': 'bucket_count', 'bucket': 0, 'count': 6,
                'numbers': [10], 'min_matches': 1}
        matched, details = eval_bucket_count_rule(self.cur, 2, rule)
        self.assertTrue(matched)
        self.assertEqual(details['bucket'], 0)
        self.assertEqual(details['count'], 6)

    def test_no_fire_when_count_differs(self):
        rule = {'type': 'bucket_count', 'bucket': 0, 'count': 3,
                'numbers': [10], 'min_matches': 1}
        matched, _ = eval_bucket_count_rule(self.cur, 2, rule)
        self.assertFalse(matched)

    def test_fires_for_correct_bucket_index(self):
        # T-1 has 0 numbers in bucket 4 (40-49)
        rule = {'type': 'bucket_count', 'bucket': 4, 'count': 0,
                'numbers': [10], 'min_matches': 1}
        matched, _ = eval_bucket_count_rule(self.cur, 2, rule)
        self.assertTrue(matched)

    def test_mixed_bucket_dist(self):
        conn, cur = _make_db({
            1: [1, 10, 20, 30, 40, 49],  # T-1: 1 per bucket (except 40-49 has 2: 40,49)
            2: [5, 6, 7, 8, 9, 15],
        })
        # bucket 4 (40-49): 40 and 49 → count=2
        rule = {'type': 'bucket_count', 'bucket': 4, 'count': 2,
                'numbers': [5], 'min_matches': 1}
        matched, _ = eval_bucket_count_rule(cur, 2, rule)
        self.assertTrue(matched)

    def test_no_previous_draw_returns_false(self):
        conn, cur = _make_db({1: [1, 2, 3, 4, 5, 6]})
        rule = {'type': 'bucket_count', 'bucket': 0, 'count': 6,
                'numbers': [1], 'min_matches': 1}
        matched, _ = eval_bucket_count_rule(cur, 1, rule)
        self.assertFalse(matched)


class TestEvalOddEvenRule(unittest.TestCase):
    def setUp(self):
        # T-1 (draw1): [1,3,5,7,9,11] → 6 odd, 0 even
        # current (draw2): [2,4,6,8,10,12]
        self.conn, self.cur = _make_db({
            1: [1, 3, 5, 7, 9, 11],
            2: [2, 4, 6, 8, 10, 12],
        })

    def test_fires_when_odd_count_matches(self):
        rule = {'type': 'odd_even', 'odd_count': 6, 'numbers': [2], 'min_matches': 1}
        matched, details = eval_odd_even_rule(self.cur, 2, rule)
        self.assertTrue(matched)
        self.assertEqual(details['odd_count'], 6)

    def test_no_fire_when_odd_count_differs(self):
        rule = {'type': 'odd_even', 'odd_count': 3, 'numbers': [2], 'min_matches': 1}
        matched, _ = eval_odd_even_rule(self.cur, 2, rule)
        self.assertFalse(matched)

    def test_three_odd_three_even(self):
        conn, cur = _make_db({
            1: [1, 2, 3, 4, 5, 6],  # T-1: 3 odd (1,3,5), 3 even (2,4,6)
            2: [7, 8, 9, 10, 11, 12],
        })
        rule = {'type': 'odd_even', 'odd_count': 3, 'numbers': [7], 'min_matches': 1}
        matched, _ = eval_odd_even_rule(cur, 2, rule)
        self.assertTrue(matched)

    def test_predicted_number_must_be_present(self):
        rule = {'type': 'odd_even', 'odd_count': 6, 'numbers': [99], 'min_matches': 1}
        matched, _ = eval_odd_even_rule(self.cur, 2, rule)
        self.assertFalse(matched)   # 99 not in draw2

    def test_no_previous_draw_returns_false(self):
        conn, cur = _make_db({1: [1, 2, 3, 4, 5, 6]})
        rule = {'type': 'odd_even', 'odd_count': 3, 'numbers': [1], 'min_matches': 1}
        matched, _ = eval_odd_even_rule(cur, 1, rule)
        self.assertFalse(matched)


class TestEvalSeedRule(unittest.TestCase):
    def setUp(self):
        # T-1 (draw1): seed=7 present; 7,14,21 are multiples → coverage=3
        # T-1 also has: 2,11,20 — digit_sum=2 matching seed=2 (cov=3) and seed=11 (cov=3)
        self.conn, self.cur = _make_db({
            1: [7, 14, 21, 2, 11, 20],  # T-1
            2: [7, 40, 22, 30, 35, 42], # current
        })

    def test_fires_seed_multiple_method(self):
        # seed=7 in T-1; multiples in T-1: {7,14,21} → cov=3 >= min_derived=2
        rule = {'type': 'seed', 'seed': 7, 'method': 'multiple', 'min_derived': 2,
                'numbers': [7], 'min_matches': 1}
        matched, details = eval_seed_rule(self.cur, 2, rule)
        self.assertTrue(matched)
        self.assertEqual(details['seed'], 7)
        self.assertEqual(details['method'], 'multiple')
        self.assertIn(7, details['covered'])
        self.assertIn(14, details['covered'])

    def test_fires_seed_last_digit_method(self):
        # seed=2 in T-1; last_digit=2 numbers: {2,12,22,32,42} ∩ T-1 = {2} → cov=1
        # min_derived=1 to pass
        rule = {'type': 'seed', 'seed': 2, 'method': 'last_digit', 'min_derived': 1,
                'numbers': [22], 'min_matches': 1}
        matched, _ = eval_seed_rule(self.cur, 2, rule)
        self.assertTrue(matched)

    def test_fires_seed_digit_sum_method(self):
        # seed=2 in T-1; digit_sum=2 → {2,11,20,29,38,47} ∩ T-1 = {2,11,20} → cov=3
        rule = {'type': 'seed', 'seed': 2, 'method': 'digit_sum', 'min_derived': 2,
                'numbers': [40], 'min_matches': 1}
        matched, details = eval_seed_rule(self.cur, 2, rule)
        self.assertTrue(matched)
        self.assertEqual(sorted(details['covered']), [2, 11, 20])

    def test_fires_seed_11_digit_sum(self):
        # seed=11 in T-1; digit_sum(11)=2 → same {2,11,20,...}; cov={2,11,20}=3 >= 2
        rule = {'type': 'seed', 'seed': 11, 'method': 'digit_sum', 'min_derived': 2,
                'numbers': [40], 'min_matches': 1}
        matched, details = eval_seed_rule(self.cur, 2, rule)
        self.assertTrue(matched)
        self.assertIn(11, details['covered'])

    def test_no_fire_when_seed_absent_from_t1(self):
        # seed=42 is NOT in T-1 (draw1)
        rule = {'type': 'seed', 'seed': 42, 'method': 'multiple', 'min_derived': 2,
                'numbers': [7], 'min_matches': 1}
        matched, _ = eval_seed_rule(self.cur, 2, rule)
        self.assertFalse(matched)

    def test_no_fire_when_coverage_below_min(self):
        # seed=7; last_digit=7 → {7,17,27,37,47} ∩ T-1={7} → cov=1 < min_derived=2
        rule = {'type': 'seed', 'seed': 7, 'method': 'last_digit', 'min_derived': 2,
                'numbers': [7], 'min_matches': 1}
        matched, _ = eval_seed_rule(self.cur, 2, rule)
        self.assertFalse(matched)

    def test_no_fire_invalid_seed(self):
        rule = {'type': 'seed', 'seed': 0, 'method': 'multiple', 'min_derived': 1,
                'numbers': [7], 'min_matches': 1}
        matched, _ = eval_seed_rule(self.cur, 2, rule)
        self.assertFalse(matched)

    def test_no_fire_invalid_method(self):
        rule = {'type': 'seed', 'seed': 7, 'method': 'unknown', 'min_derived': 1,
                'numbers': [7], 'min_matches': 1}
        matched, _ = eval_seed_rule(self.cur, 2, rule)
        self.assertFalse(matched)

    def test_no_fire_when_predicted_number_absent(self):
        # Seed fires but predicted number 99 not in draw2
        rule = {'type': 'seed', 'seed': 7, 'method': 'multiple', 'min_derived': 2,
                'numbers': [99], 'min_matches': 1}
        matched, _ = eval_seed_rule(self.cur, 2, rule)
        self.assertFalse(matched)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        sys.argv.pop(1)
        unittest.main()
    else:
        main()
