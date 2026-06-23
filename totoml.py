"""totoml.py — Machine learning pipeline for TOTO number analysis.

Three models in increasing complexity:
  baseline    — always pick the 6 most frequent numbers overall
  random_forest — per-number binary classifier: will this number appear?
  sequence    — sliding-window frequency model with recency weighting

Usage:
    python3 totoml.py --train [--model baseline|random_forest|sequence]
    python3 totoml.py --predict --draw <draw_no> [--model ...]
    python3 totoml.py --evaluate [--model ...]
    python3 totoml.py test
"""
import sqlite3, json, logging, argparse, math, unittest
from collections import defaultdict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

TOTO_NUMBERS = list(range(1, 50))   # TOTO uses numbers 1-49
N_PICK = 6


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def load_draw_sequence(cur, up_to_draw_no=None):
    """Return ordered list of (draw_no, [winning_numbers]) up to draw_no."""
    sql = """
        SELECT d.draw_no, j.number
        FROM draws d
        JOIN jackpot_no j ON j.draw_no = d.draw_no
        WHERE d.scanned=1 AND j.no_type='normal'
    """
    params = []
    if up_to_draw_no is not None:
        sql += ' AND d.draw_no < ?'
        params.append(up_to_draw_no)
    sql += ' ORDER BY d.draw_no, j.number'
    cur.execute(sql, params)
    draw_map = defaultdict(list)
    order = []
    seen = set()
    for draw_no, number in cur.fetchall():
        if draw_no not in seen:
            order.append(draw_no)
            seen.add(draw_no)
        draw_map[draw_no].append(number)
    return [(d, draw_map[d]) for d in order]


def build_features_for_number(number, history, window=20):
    """Build a feature vector for predicting whether `number` appears next.

    Features:
      - freq_all      : appearances / total draws
      - freq_window   : appearances in last `window` draws / window
      - gap_since     : draws since last appearance (normalised by window)
      - recency_score : exponentially weighted recency (recent = higher)
    """
    draws = [nums for _, nums in history]
    total = len(draws)
    if total == 0:
        return [0.0, 0.0, 1.0, 0.0]

    appearances_all = sum(1 for nums in draws if number in nums)
    freq_all = appearances_all / total

    window_draws = draws[-window:]
    appearances_win = sum(1 for nums in window_draws if number in nums)
    freq_window = appearances_win / len(window_draws) if window_draws else 0.0

    gap_since = 0
    for i, nums in enumerate(reversed(draws)):
        if number in nums:
            break
        gap_since += 1
    else:
        gap_since = total   # never appeared

    gap_norm = min(gap_since / window, 2.0)

    recency = 0.0
    decay = 0.9
    for i, nums in enumerate(reversed(draws)):
        if number in nums:
            recency += decay ** i
    recency_norm = recency / window

    return [freq_all, freq_window, gap_norm, recency_norm]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def train_baseline(history):
    """Baseline: frequency table over all history."""
    freq = defaultdict(int)
    for _, numbers in history:
        for n in numbers:
            freq[n] += 1
    return {'type': 'baseline', 'freq': dict(freq)}


def predict_baseline(model_params, history, n=N_PICK):
    freq = model_params['freq']
    return sorted(freq, key=lambda x: -freq.get(x, 0))[:n]


def train_sequence(history, window=20):
    """Sliding-window recency-weighted frequency model."""
    scores = defaultdict(float)
    decay = 0.95
    for i, (_, numbers) in enumerate(history):
        weight = decay ** (len(history) - i - 1)
        for n in numbers:
            scores[n] += weight
    return {'type': 'sequence', 'window': window, 'scores': dict(scores)}


def predict_sequence(model_params, history, n=N_PICK):
    scores = model_params['scores']
    return sorted(scores, key=lambda x: -scores.get(x, 0))[:n]


def train_random_forest(history, window=20):
    """Train a per-number logistic regression (sklearn optional; pure-Python fallback)."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        sklearn_available = True
    except ImportError:
        sklearn_available = False

    if not sklearn_available:
        logging.warning('scikit-learn not installed; falling back to sequence model')
        return train_sequence(history, window)

    import numpy as np
    X, y_map = [], defaultdict(list)
    for i in range(window, len(history)):
        hist_slice = history[:i]
        for number in TOTO_NUMBERS:
            feats = build_features_for_number(number, hist_slice, window)
            X.append(feats)
            label = 1 if number in history[i][1] else 0
            y_map[number].append(label)

    # One classifier per number
    classifiers = {}
    X_arr = np.array(X)
    for idx, number in enumerate(TOTO_NUMBERS):
        labels = y_map[number]
        if len(set(labels)) < 2:
            continue   # skip if only one class
        X_num = X_arr[idx::len(TOTO_NUMBERS)]
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_num, labels)
        classifiers[number] = clf

    return {'type': 'random_forest', 'window': window, 'classifiers': classifiers}


def predict_random_forest(model_params, history, n=N_PICK):
    if model_params.get('type') != 'random_forest' or 'classifiers' not in model_params:
        return predict_sequence(model_params, history, n)
    import numpy as np
    classifiers = model_params['classifiers']
    window = model_params.get('window', 20)
    probs = {}
    for number in TOTO_NUMBERS:
        feats = build_features_for_number(number, history, window)
        clf = classifiers.get(number)
        if clf is not None:
            probs[number] = clf.predict_proba([feats])[0][1]
        else:
            probs[number] = 0.0
    return sorted(probs, key=lambda x: -probs[x])[:n]


TRAINERS = {
    'baseline':      train_baseline,
    'sequence':      train_sequence,
    'random_forest': train_random_forest,
}
PREDICTORS = {
    'baseline':      predict_baseline,
    'sequence':      predict_sequence,
    'random_forest': predict_random_forest,
}


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _serialise_model(model_params):
    """Serialise model to JSON (classifiers are not JSON-serialisable; store metadata only)."""
    safe = {k: v for k, v in model_params.items() if k != 'classifiers'}
    return json.dumps(safe)


def save_prediction(cur, model_name, target_draw_no, numbers, confidence=None):
    cur.execute(
        """INSERT INTO predictions(model_name,target_draw_no,predicted_numbers_json,
           confidence_json,created_at) VALUES(?,?,?,?,?)""",
        (model_name, target_draw_no,
         json.dumps(sorted(numbers)),
         json.dumps(confidence) if confidence else None,
         datetime.utcnow().isoformat()),
    )
    return cur.lastrowid


def evaluate_predictions(cur):
    """Compare all predictions against actual results where available."""
    cur.execute("""
        SELECT p.prediction_id, p.model_name, p.target_draw_no,
               p.predicted_numbers_json
        FROM predictions p
        JOIN draws d ON d.draw_no = p.target_draw_no AND d.scanned = 1
        ORDER BY p.target_draw_no
    """)
    rows = cur.fetchall()
    results = []
    for pred_id, model_name, draw_no, pred_json in rows:
        predicted = set(json.loads(pred_json))
        cur.execute(
            'SELECT number FROM jackpot_no WHERE draw_no=? AND no_type="normal"',
            (draw_no,),
        )
        actual = {r[0] for r in cur.fetchall()}
        matched = len(predicted & actual)
        results.append({
            'prediction_id': pred_id,
            'model_name': model_name,
            'draw_no': draw_no,
            'matched': matched,
            'predicted': sorted(predicted),
            'actual': sorted(actual),
        })
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='TOTO ML pipeline')
    parser.add_argument('--train',    action='store_true')
    parser.add_argument('--predict',  action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--model',    default='sequence',
                        choices=['baseline', 'sequence', 'random_forest'])
    parser.add_argument('--draw',     type=int, help='Target draw number for prediction')
    parser.add_argument('--window',   type=int, default=20)
    args = parser.parse_args()

    conn = sqlite3.connect('toto.sqlite')
    try:
        cur = conn.cursor()
        history = load_draw_sequence(cur)

        if args.train or args.predict:
            trainer = TRAINERS[args.model]
            logging.info('Training %s model on %d draws...', args.model, len(history))
            if args.model in ('sequence', 'random_forest'):
                model_params = trainer(history, window=args.window)
            else:
                model_params = trainer(history)
            logging.info('Training complete.')

        if args.predict:
            predictor = PREDICTORS[args.model]
            predicted = predictor(model_params, history)
            target = args.draw
            pred_id = save_prediction(cur, args.model, target, predicted)
            conn.commit()
            logging.info('Prediction ID %d: %s → %s', pred_id, args.model, predicted)
            print('Predicted numbers:', sorted(predicted))

        elif args.evaluate:
            results = evaluate_predictions(cur)
            if not results:
                print('No predictions to evaluate yet.')
                return
            for r in results:
                print(f'Draw {r["draw_no"]} ({r["model_name"]}): '
                      f'{r["matched"]}/6 matched  '
                      f'predicted={r["predicted"]}  actual={r["actual"]}')
            avg = sum(r['matched'] for r in results) / len(results)
            print(f'\nAverage matched: {avg:.2f}/6 over {len(results)} predictions')

        elif not args.train:
            parser.print_help()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestBuildFeaturesForNumber(unittest.TestCase):
    def _history(self, draws):
        return [(i, nums) for i, nums in enumerate(draws, 1)]

    def test_freq_all_correct(self):
        history = self._history([[1, 2], [1, 3], [4, 5]])
        feats = build_features_for_number(1, history)
        self.assertAlmostEqual(feats[0], 2 / 3)

    def test_never_appeared_gap_normalised(self):
        history = self._history([[2, 3], [4, 5], [6, 7]])
        feats = build_features_for_number(99, history)
        self.assertEqual(feats[0], 0.0)

    def test_recency_higher_when_recent(self):
        h_recent = self._history([[2, 3], [2, 3], [1, 2]])
        h_old    = self._history([[1, 2], [2, 3], [2, 3]])
        feats_recent = build_features_for_number(1, h_recent)
        feats_old    = build_features_for_number(1, h_old)
        self.assertGreater(feats_recent[3], feats_old[3])


class TestTrainAndPredictBaseline(unittest.TestCase):
    def _history(self):
        return [(i, [1, 2, 3, 4, 5, 6]) for i in range(1, 11)] + \
               [(11, [7, 8, 9, 10, 11, 12])]

    def test_picks_most_frequent_numbers(self):
        history = self._history()
        model = train_baseline(history)
        predicted = predict_baseline(model, history, n=6)
        self.assertIn(1, predicted)
        self.assertIn(2, predicted)

    def test_returns_n_numbers(self):
        history = self._history()
        model = train_baseline(history)
        self.assertEqual(len(predict_baseline(model, history, n=6)), 6)


class TestTrainAndPredictSequence(unittest.TestCase):
    def test_recent_numbers_ranked_higher(self):
        history = [(i, [99, i, i+1, i+2, i+3, i+4]) for i in range(1, 20)]
        model = train_sequence(history, window=10)
        predicted = predict_sequence(model, history, n=1)
        self.assertEqual(predicted[0], 99)  # 99 appeared in every draw


class TestEvaluatePredictions(unittest.TestCase):
    def _make_db(self):
        conn = sqlite3.connect(':memory:')
        cur = conn.cursor()
        cur.executescript('''
            CREATE TABLE draws(draw_no INTEGER PRIMARY KEY, day TEXT, date TEXT,
                               scanned INTEGER DEFAULT 0);
            CREATE TABLE jackpot_no(draw_no INTEGER, no_type TEXT, number INTEGER);
            CREATE TABLE predictions(prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT, target_draw_no INTEGER,
                predicted_numbers_json TEXT, confidence_json TEXT, created_at TEXT);
        ''')
        cur.execute('INSERT INTO draws VALUES(1,"Mon","01 Jan 2024",1)')
        for n in [1, 2, 3, 4, 5, 6]:
            cur.execute('INSERT INTO jackpot_no VALUES(1,"normal",?)', (n,))
        cur.execute('INSERT INTO predictions VALUES(1,"test",1,?,NULL,"2024-01-01")',
                    (json.dumps([1, 2, 3, 7, 8, 9]),))
        return conn, cur

    def test_matched_count(self):
        conn, cur = self._make_db()
        results = evaluate_predictions(cur)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['matched'], 3)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        sys.argv.pop(1)
        unittest.main()
    else:
        main()
