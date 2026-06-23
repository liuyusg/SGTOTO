from bs4 import BeautifulSoup
import sqlite3, re, logging, base64, unittest, os
import requests
from unittest.mock import MagicMock

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

BASE_URL = 'https://www.singaporepools.com.sg/en/product/sr/Pages/toto_results.aspx'
DRAW_LIST_URL = 'https://www.singaporepools.com.sg/DataFileArchive/Lottery/Output/toto_result_draw_list_en.html'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36'
}


def fetch_draw_list(session):
    """Fetch the pre-generated draw list HTML that contains all draw numbers."""
    resp = session.get(DRAW_LIST_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def fetch_draw(session, draw_no):
    """Fetch the results page for a specific draw number."""
    sppl = base64.b64encode('DrawNumber={}'.format(draw_no).encode()).decode()
    resp = session.get(BASE_URL, params={'sppl': sppl}, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_draw_numbers(html):
    """Return list of draw numbers found in the draw list HTML."""
    return re.findall(r'value="(\d{3,4})"', html)


def parse_draw_page(html):
    """Parse a draw results page.

    Returns (day, date_str, numbers, additional) or None if drawDate is missing.
    numbers is a list of the 6 winning number strings.
    additional is a string or None.
    """
    soup = BeautifulSoup(html, 'html.parser')
    date_cells = soup.select('th[class="drawDate"]')
    if not date_cells:
        return None
    date_parts = date_cells[0].getText().split(', ')
    day = date_parts[0]
    date_str = date_parts[1]

    numbers = [td.getText() for td in soup.select('td[width="16%"]')]

    additional = None
    additional_cells = soup.select('td[class="additional"]')
    if additional_cells:
        additional = additional_cells[0].getText()

    return day, date_str, numbers, additional


def parse_locations(html):
    """Parse Group 1 winner location entries from a draw results page.

    Returns a list of tuples: (raw_text, place, address, draw_type, system).
    Returns [] if Group 1 has no winner.
    """
    if 'Group 1 has no winner' in html:
        return []

    truncated = html[:html.find('Group 2 winning tickets')]
    soup = BeautifulSoup(truncated, 'html.parser')
    results = []
    for t in soup.select('li'):
        try:
            raw = t.getText()
            a = raw.replace('\n', '').replace('  ', '').replace(' )', '').split('( ')
            location = a[0].split(' - ')
            place = location[0]
            address = location[1]
            if a[1].find('QuickPick') == -1:
                drawtype = re.search(r'1\s(.+)\sE', a[1])
                if drawtype is None:
                    logging.warning('Regex did not match, raw: %s', a[1])
                    continue
                draw = 'No QuickPick'
                system = drawtype.group(1)
            else:
                drawtype = re.search(r'1\s(\w+)\s(.+)\sE', a[1])
                if drawtype is None:
                    logging.warning('Regex did not match, raw: %s', a[1])
                    continue
                draw = drawtype.group(1)
                system = drawtype.group(2)
            results.append((raw, place, address, draw, system))
        except Exception as e:
            logging.warning('Skipping malformed location entry: %s', e)
    return results


def html_path(draw_no, history_dir='history'):
    """Return the file path for storing the raw HTML of a draw.

    Follows the naming convention already used in the history/ folder:
      history/sppl=<base64(DrawNumber=NNNN)>.txt
    """
    sppl = base64.b64encode('DrawNumber={}'.format(draw_no).encode()).decode()
    return os.path.join(history_dir, 'sppl={}.txt'.format(sppl))


def save_html(draw_no, html, history_dir='history'):
    """Save raw HTML to the history folder. No-op if the file already exists."""
    os.makedirs(history_dir, exist_ok=True)
    path = html_path(draw_no, history_dir)
    if not os.path.exists(path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        logging.info('Saved HTML: %s', path)
    return path


def is_valid_draw_html(html):
    """Return True if html looks like a complete draw result page."""
    return bool(html) and 'drawDate' in html


def load_html(draw_no, history_dir='history'):
    """Load HTML from the history folder.

    Returns the cached content if it passes the validity check.
    Deletes the file and returns None if the cached content is invalid,
    so that a fresh download can replace it.
    """
    path = html_path(draw_no, history_dir)
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            content = f.read()
        if is_valid_draw_html(content):
            return content
        os.remove(path)
        logging.warning('Removed invalid cached file: %s', path)
    return None


def parse_prize_table(html):
    """Parse the winning shares table for all 7 prize groups.

    Returns a list of (prize_group, prize_amount_cents, winner_count).
    prize_amount_cents is an integer (dollars * 100), or None if unparseable.
    """
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.select_one('table.tableWinningShares')
    if not table:
        return []
    results = []
    for row in table.select('tbody tr'):
        cells = row.select('td')
        if len(cells) < 3:
            continue
        group_text = cells[0].getText().strip()
        amount_text = cells[1].getText().strip()
        count_text = cells[2].getText().strip().replace(',', '')
        group_match = re.search(r'\d+', group_text)
        if group_match is None:
            continue
        prize_group = int(group_match.group())
        amount_match = re.search(r'[\d,]+', amount_text.replace('$', ''))
        prize_cents = int(amount_match.group().replace(',', '')) * 100 if amount_match else None
        winner_count = int(count_text) if count_text.isdigit() else None
        results.append((prize_group, prize_cents, winner_count))
    return results


def parse_all_winner_locations(html):
    """Parse winner location entries for ALL prize groups.

    Returns a list of tuples: (prize_group, raw_text, place, address, draw_type, system).
    """
    results = []
    pattern = re.compile(r'Group (\d+) winning tickets sold at', re.IGNORECASE)
    segments = pattern.split(html)
    # segments = [pre_text, group_no, segment_html, group_no, segment_html, ...]
    i = 1
    while i < len(segments) - 1:
        try:
            prize_group = int(segments[i])
            segment_html = segments[i + 1]
            # cut at next group marker or end
            soup = BeautifulSoup(segment_html, 'html.parser')
            for t in soup.select('li'):
                try:
                    raw = t.getText()
                    a = raw.replace('\n', '').replace('  ', '').replace(' )', '').split('( ')
                    if len(a) < 2:
                        continue
                    location = a[0].split(' - ')
                    if len(location) < 2:
                        continue
                    place = location[0]
                    address = location[1]
                    if a[1].find('QuickPick') == -1:
                        drawtype = re.search(r'1\s(.+)\sE', a[1])
                        if drawtype is None:
                            continue
                        draw = 'No QuickPick'
                        system = drawtype.group(1)
                    else:
                        drawtype = re.search(r'1\s(\w+)\s(.+)\sE', a[1])
                        if drawtype is None:
                            continue
                        draw = drawtype.group(1)
                        system = drawtype.group(2)
                    results.append((prize_group, raw, place, address, draw, system))
                except Exception as e:
                    logging.warning('Skipping malformed winner entry: %s', e)
        except (ValueError, IndexError):
            pass
        i += 2
    return results


# ---------------------------------------------------------------------------
# Database import helpers
# ---------------------------------------------------------------------------

def _import_draw(cur, draw_no, html):
    """Parse html and upsert all draw data for draw_no.

    Returns True on success, False if the page has no drawDate.
    """
    result = parse_draw_page(html)
    if result is None:
        logging.warning('No drawDate found for draw %s, skipping', draw_no)
        return False
    day, date1, numbers, additional = result

    cur.execute("INSERT OR IGNORE INTO draws(draw_no) VALUES(?)", (draw_no,))
    cur.execute("UPDATE draws SET day=?,date=?,scanned=1 WHERE draw_no=?", (day, date1, draw_no))
    logging.info('Draw %s: %s %s', draw_no, day, date1)

    for number in numbers:
        cur.execute("INSERT OR IGNORE INTO jackpot_no(draw_no,no_type,number) VALUES(?,?,?)",
                    (draw_no, 'normal', number))
    if additional:
        cur.execute("INSERT OR IGNORE INTO jackpot_no(draw_no,no_type,number) VALUES(?,?,?)",
                    (draw_no, 'additional', additional))

    for prize_group, prize_cents, winner_count in parse_prize_table(html):
        cur.execute("INSERT OR IGNORE INTO draw_prizes(draw_no,prize_group,prize_amount,winner_count) VALUES(?,?,?,?)",
                    (draw_no, prize_group, prize_cents, winner_count))

    for raw, place, address, draw, system in parse_locations(html):
        logging.info('Location: %s  draw: %s  system: %s', place, draw, system)
        cur.execute("INSERT OR IGNORE INTO place(draw_no,raw_data,location,address,quickpick,system) VALUES(?,?,?,?,?,?)",
                    (draw_no, raw, place, address, draw, system))

    for prize_group, raw, place, address, draw, system in parse_all_winner_locations(html):
        cur.execute("INSERT OR IGNORE INTO winners(draw_no,prize_group,raw_data,location,address,quickpick,system) VALUES(?,?,?,?,?,?,?)",
                    (draw_no, prize_group, raw, place, address, draw, system))
    return True


def load_history(db_path='toto.sqlite', history_dir='history'):
    """Import every cached HTML file in history_dir into the database.

    Files named ``sppl=<base64>.txt`` are decoded to recover the draw number.
    Already-scanned draws are skipped.  Returns the count of newly imported draws.
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        if not os.path.isdir(history_dir):
            logging.warning('History directory not found: %s', history_dir)
            return 0

        imported = 0
        for fname in sorted(os.listdir(history_dir)):
            if not (fname.startswith('sppl=') and fname.endswith('.txt')):
                continue
            sppl = fname[5:-4]          # strip 'sppl=' prefix and '.txt' suffix
            try:
                decoded = base64.b64decode(sppl).decode()
                m = re.match(r'DrawNumber=(\d+)', decoded)
                if not m:
                    continue
                draw_no = int(m.group(1))
            except Exception:
                continue

            # skip draws already in the database and marked scanned
            cur.execute("SELECT scanned FROM draws WHERE draw_no=?", (draw_no,))
            row = cur.fetchone()
            if row and row[0] == 1:
                continue

            path = os.path.join(history_dir, fname)
            with open(path, encoding='utf-8') as f:
                html = f.read()

            if not is_valid_draw_html(html):
                logging.warning('Skipping invalid cached file: %s', fname)
                continue

            if _import_draw(cur, draw_no, html):
                imported += 1
                conn.commit()

        logging.info('Loaded %d new draws from %s', imported, history_dir)
        return imported
    finally:
        conn.close()


VALID_DAYS   = {'Mon', 'Thu'}
MONTH_ABBRS  = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
                 'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}


def manual_add_draw(draw_no, day, date_str, numbers, additional=None, db_path='toto.sqlite'):
    """Manually insert a draw result into the database.

    draw_no   : positive int   (e.g. 4193)
    day       : 'Mon' or 'Thu'
    date_str  : 'DD Mon YYYY'  (e.g. '23 Jun 2026')
    numbers   : list of 6 distinct ints in 1-49
    additional: optional int 1-49, must not duplicate a winning number
    db_path   : SQLite database path

    Returns a dict with 'draw_no', 'day', 'date', 'numbers', 'additional'.
    Raises ValueError for invalid input.
    """
    # --- validate draw_no ---
    if not isinstance(draw_no, int) or draw_no <= 0:
        raise ValueError(f'draw_no must be a positive integer, got {draw_no!r}')

    # --- validate day ---
    if day not in VALID_DAYS:
        raise ValueError(f'day must be Mon or Thu, got {day!r}')

    # --- validate date_str format: 'DD Mon YYYY' ---
    parts = date_str.strip().split()
    if len(parts) != 3 or not parts[0].isdigit() or parts[1] not in MONTH_ABBRS or not parts[2].isdigit():
        raise ValueError(f'date must be in DD Mon YYYY format (e.g. 23 Jun 2026), got {date_str!r}')
    dom, year = int(parts[0]), int(parts[2])
    month = MONTH_ABBRS[parts[1]]
    try:
        from datetime import date as _date
        _date(year, month, dom)   # raises ValueError for impossible dates
    except ValueError:
        raise ValueError(f'Invalid calendar date: {date_str!r}')

    # --- validate numbers ---
    nums = list(numbers)
    if len(nums) != 6:
        raise ValueError(f'Exactly 6 winning numbers required, got {len(nums)}')
    for n in nums:
        if not isinstance(n, int) or not (1 <= n <= 49):
            raise ValueError(f'Each winning number must be an integer 1-49, got {n!r}')
    if len(set(nums)) != 6:
        raise ValueError(f'Winning numbers must be distinct, got {nums}')

    # --- validate additional ---
    if additional is not None:
        if not isinstance(additional, int) or not (1 <= additional <= 49):
            raise ValueError(f'Additional number must be an integer 1-49, got {additional!r}')
        if additional in nums:
            raise ValueError(f'Additional number {additional} duplicates a winning number')

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute('SELECT scanned FROM draws WHERE draw_no=?', (draw_no,))
        existing = cur.fetchone()
        if existing and existing[0] == 1:
            raise ValueError(
                f'Draw {draw_no} already exists in the database with scanned=1. '
                'Delete it first if you want to re-enter.'
            )
        cur.execute('INSERT OR IGNORE INTO draws(draw_no) VALUES(?)', (draw_no,))
        cur.execute('UPDATE draws SET day=?, date=?, scanned=1 WHERE draw_no=?',
                    (day, date_str.strip(), draw_no))
        for n in sorted(nums):
            cur.execute('INSERT OR IGNORE INTO jackpot_no(draw_no,no_type,number) VALUES(?,?,?)',
                        (draw_no, 'normal', n))
        if additional is not None:
            cur.execute('INSERT OR IGNORE INTO jackpot_no(draw_no,no_type,number) VALUES(?,?,?)',
                        (draw_no, 'additional', additional))
        conn.commit()
        logging.info('Manually added draw %d (%s %s): %s  additional=%s',
                     draw_no, day, date_str, sorted(nums), additional)
    finally:
        conn.close()

    return {'draw_no': draw_no, 'day': day, 'date': date_str.strip(),
            'numbers': sorted(nums), 'additional': additional}


def main(db_path='toto.sqlite', history_dir='history'):
    import argparse
    parser = argparse.ArgumentParser(description='Scrape TOTO draw results')
    parser.add_argument('--load-history', action='store_true',
                        help='Import all cached HTML files from history/ into the database')
    parser.add_argument('--draw',       type=int,   metavar='N',
                        help='Draw number for manual entry (e.g. 4193)')
    parser.add_argument('--day',        metavar='Mon|Thu',
                        help='Draw day for manual entry')
    parser.add_argument('--date',       metavar='DD Mon YYYY',
                        help='Draw date for manual entry (e.g. "23 Jun 2026")')
    parser.add_argument('--numbers',    metavar='N1,N2,N3,N4,N5,N6',
                        help='Comma-separated 6 winning numbers for manual entry')
    parser.add_argument('--additional', type=int,   metavar='N',
                        help='Optional additional (7th) number for manual entry')
    args, _ = parser.parse_known_args()

    if args.load_history:
        load_history(db_path=db_path, history_dir=history_dir)
        return

    if args.draw or args.day or args.date or args.numbers:
        missing = [f for f, v in [('--draw', args.draw), ('--day', args.day),
                                   ('--date', args.date), ('--numbers', args.numbers)] if not v]
        if missing:
            parser.error(f'Manual entry requires all of: {" ".join(missing)}')
        try:
            nums = [int(x.strip()) for x in args.numbers.split(',')]
            result = manual_add_draw(
                draw_no=args.draw,
                day=args.day,
                date_str=args.date,
                numbers=nums,
                additional=args.additional,
                db_path=db_path,
            )
            print(f"Added draw {result['draw_no']} ({result['day']} {result['date']}): "
                  f"{result['numbers']}  additional={result['additional']}")
        except ValueError as e:
            print(f'Error: {e}')
        return

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        session = requests.Session()

        # get list of toto draw numbers from pre-generated draw list file
        draw_list_html = fetch_draw_list(session)
        alltoto = parse_draw_numbers(draw_list_html)
        for draw_no in alltoto:
            cur.execute("INSERT OR IGNORE INTO draws(draw_no) VALUES(?)", (draw_no,))
        conn.commit()
        logging.info('Found %d draw numbers in draw list', len(alltoto))

        # extract draw numbers not scanned
        cur.execute("SELECT draw_no FROM draws WHERE scanned=0")
        alltoto = cur.fetchall()

        for row in alltoto:
            for draw_no in row:
                # use cached HTML if available, otherwise fetch and save
                html = load_html(draw_no)
                if html is None:
                    html = fetch_draw(session, draw_no)
                    save_html(draw_no, html)

                if _import_draw(cur, draw_no, html):
                    conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestManualAddDraw(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.db = tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False)
        self.db_path = self.db.name
        self.db.close()
        # Bootstrap schema from toto.sqlite so the tables exist
        import shutil
        shutil.copy('toto.sqlite', self.db_path)

    def tearDown(self):
        import os
        os.unlink(self.db_path)

    def _query(self, sql, *args):
        conn = sqlite3.connect(self.db_path)
        try:
            return conn.cursor().execute(sql, args).fetchall()
        finally:
            conn.close()

    # --- success cases ---

    def test_adds_draw_and_numbers(self):
        manual_add_draw(9999, 'Mon', '23 Jun 2026', [1, 7, 15, 22, 35, 42],
                        db_path=self.db_path)
        rows = self._query("SELECT day, date, scanned FROM draws WHERE draw_no=9999")
        self.assertEqual(rows, [('Mon', '23 Jun 2026', 1)])
        nums = sorted(r[0] for r in self._query(
            "SELECT number FROM jackpot_no WHERE draw_no=9999 AND no_type='normal'"))
        self.assertEqual(nums, [1, 7, 15, 22, 35, 42])

    def test_adds_additional_number(self):
        manual_add_draw(9998, 'Thu', '19 Jun 2026', [2, 8, 14, 21, 30, 45],
                        additional=11, db_path=self.db_path)
        rows = self._query(
            "SELECT number FROM jackpot_no WHERE draw_no=9998 AND no_type='additional'")
        self.assertEqual(rows, [(11,)])

    def test_no_additional_when_omitted(self):
        manual_add_draw(9997, 'Mon', '16 Jun 2026', [3, 9, 17, 25, 33, 41],
                        db_path=self.db_path)
        rows = self._query(
            "SELECT number FROM jackpot_no WHERE draw_no=9997 AND no_type='additional'")
        self.assertEqual(rows, [])

    def test_returns_dict_with_correct_fields(self):
        result = manual_add_draw(9996, 'Thu', '12 Jun 2026', [5, 10, 20, 30, 40, 49],
                                 additional=7, db_path=self.db_path)
        self.assertEqual(result['draw_no'], 9996)
        self.assertEqual(result['day'], 'Thu')
        self.assertEqual(result['date'], '12 Jun 2026')
        self.assertEqual(result['numbers'], [5, 10, 20, 30, 40, 49])
        self.assertEqual(result['additional'], 7)

    def test_numbers_stored_sorted(self):
        manual_add_draw(9995, 'Mon', '09 Jun 2026', [42, 7, 1, 35, 22, 15],
                        db_path=self.db_path)
        nums = [r[0] for r in self._query(
            "SELECT number FROM jackpot_no WHERE draw_no=9995 AND no_type='normal' ORDER BY number")]
        self.assertEqual(nums, [1, 7, 15, 22, 35, 42])

    # --- validation errors ---

    def test_rejects_invalid_day(self):
        with self.assertRaises(ValueError):
            manual_add_draw(9994, 'Tue', '23 Jun 2026', [1, 2, 3, 4, 5, 6],
                            db_path=self.db_path)

    def test_rejects_bad_date_format(self):
        with self.assertRaises(ValueError):
            manual_add_draw(9993, 'Mon', '2026-06-23', [1, 2, 3, 4, 5, 6],
                            db_path=self.db_path)

    def test_rejects_impossible_date(self):
        with self.assertRaises(ValueError):
            manual_add_draw(9992, 'Mon', '31 Apr 2026', [1, 2, 3, 4, 5, 6],
                            db_path=self.db_path)

    def test_rejects_wrong_count_of_numbers(self):
        with self.assertRaises(ValueError):
            manual_add_draw(9991, 'Mon', '23 Jun 2026', [1, 2, 3, 4, 5],
                            db_path=self.db_path)

    def test_rejects_number_out_of_range(self):
        with self.assertRaises(ValueError):
            manual_add_draw(9990, 'Mon', '23 Jun 2026', [0, 2, 3, 4, 5, 6],
                            db_path=self.db_path)
        with self.assertRaises(ValueError):
            manual_add_draw(9989, 'Mon', '23 Jun 2026', [1, 2, 3, 4, 5, 50],
                            db_path=self.db_path)

    def test_rejects_duplicate_numbers(self):
        with self.assertRaises(ValueError):
            manual_add_draw(9988, 'Mon', '23 Jun 2026', [1, 1, 3, 4, 5, 6],
                            db_path=self.db_path)

    def test_rejects_additional_duplicating_winning_number(self):
        with self.assertRaises(ValueError):
            manual_add_draw(9987, 'Mon', '23 Jun 2026', [1, 2, 3, 4, 5, 6],
                            additional=3, db_path=self.db_path)

    def test_rejects_additional_out_of_range(self):
        with self.assertRaises(ValueError):
            manual_add_draw(9986, 'Mon', '23 Jun 2026', [1, 2, 3, 4, 5, 6],
                            additional=50, db_path=self.db_path)

    def test_rejects_nonpositive_draw_no(self):
        with self.assertRaises(ValueError):
            manual_add_draw(0, 'Mon', '23 Jun 2026', [1, 2, 3, 4, 5, 6],
                            db_path=self.db_path)

    def test_rejects_overwrite_of_existing_draw(self):
        manual_add_draw(9985, 'Mon', '23 Jun 2026', [1, 2, 3, 4, 5, 6],
                        db_path=self.db_path)
        with self.assertRaises(ValueError):
            manual_add_draw(9985, 'Thu', '26 Jun 2026', [7, 8, 9, 10, 11, 12],
                            db_path=self.db_path)


class TestHtmlPath(unittest.TestCase):
    def test_known_draw_number(self):
        # sppl=RHJhd051bWJlcj0zOTYz is the known encoding of DrawNumber=3963
        path = html_path(3963, history_dir='history')
        self.assertEqual(path, 'history/sppl=RHJhd051bWJlcj0zOTYz.txt')

    def test_different_draw_number(self):
        path = html_path(3962, history_dir='history')
        sppl = base64.b64encode(b'DrawNumber=3962').decode()
        self.assertEqual(path, 'history/sppl={}.txt'.format(sppl))

    def test_custom_history_dir(self):
        path = html_path(100, history_dir='/tmp/myhistory')
        self.assertTrue(path.startswith('/tmp/myhistory/'))
        self.assertTrue(path.endswith('.txt'))


class TestSaveAndLoadHtml(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    VALID_HTML = "<th class='drawDate'>Mon, 08 Apr 2024</th>"

    def test_save_creates_file(self):
        save_html(3963, self.VALID_HTML, history_dir=self.tmp)
        path = html_path(3963, history_dir=self.tmp)
        self.assertTrue(os.path.exists(path))

    def test_load_returns_content(self):
        save_html(3963, self.VALID_HTML, history_dir=self.tmp)
        content = load_html(3963, history_dir=self.tmp)
        self.assertEqual(content, self.VALID_HTML)

    def test_load_returns_none_when_missing(self):
        self.assertIsNone(load_html(9999, history_dir=self.tmp))

    def test_save_does_not_overwrite_existing(self):
        save_html(3963, self.VALID_HTML, history_dir=self.tmp)
        save_html(3963, '<th class="drawDate">overwrite</th>', history_dir=self.tmp)
        content = load_html(3963, history_dir=self.tmp)
        self.assertEqual(content, self.VALID_HTML)

    def test_creates_history_dir_if_missing(self):
        import shutil
        nested = os.path.join(self.tmp, 'new_subdir')
        save_html(1, self.VALID_HTML, history_dir=nested)
        self.assertTrue(os.path.isdir(nested))

    def test_load_returns_none_for_invalid_content(self):
        path = html_path(3963, history_dir=self.tmp)
        os.makedirs(self.tmp, exist_ok=True)
        with open(path, 'w') as f:
            f.write('<html>no draw date here</html>')
        self.assertIsNone(load_html(3963, history_dir=self.tmp))
        self.assertFalse(os.path.exists(path))  # invalid file deleted

    def test_load_returns_none_for_empty_file(self):
        path = html_path(3963, history_dir=self.tmp)
        os.makedirs(self.tmp, exist_ok=True)
        with open(path, 'w') as f:
            f.write('')
        self.assertIsNone(load_html(3963, history_dir=self.tmp))
        self.assertFalse(os.path.exists(path))


class TestParseDrawNumbers(unittest.TestCase):
    def test_extracts_four_digit_draw_numbers(self):
        html = '<option value="3963">Draw No. 3963</option><option value="3962">Draw No. 3962</option>'
        self.assertEqual(parse_draw_numbers(html), ['3963', '3962'])

    def test_extracts_three_digit_draw_numbers(self):
        html = '<option value="999">Draw No. 999</option>'
        self.assertEqual(parse_draw_numbers(html), ['999'])

    def test_ignores_five_digit_values(self):
        html = '<option value="12345">foo</option>'
        self.assertEqual(parse_draw_numbers(html), [])

    def test_ignores_non_numeric_values(self):
        html = '<option value="abc">foo</option>'
        self.assertEqual(parse_draw_numbers(html), [])

    def test_empty_html_returns_empty_list(self):
        self.assertEqual(parse_draw_numbers(''), [])


class TestParseDrawPage(unittest.TestCase):
    SAMPLE_HTML = """
    <table><tr>
      <th width='50%' class='drawDate'>Mon, 08 Apr 2024</th>
    </tr></table>
    <table><tbody><tr>
      <td width='16%' class='win1'>12</td>
      <td width='16%' class='win2'>23</td>
      <td width='16%' class='win3'>24</td>
      <td width='16%' class='win4'>34</td>
      <td width='16%' class='win5'>43</td>
      <td width='16%' class='win6'>46</td>
    </tr></tbody></table>
    <table><tbody><tr>
      <td class='additional'>42</td>
    </tr></tbody></table>
    """

    def test_parse_day(self):
        day, _, _, _ = parse_draw_page(self.SAMPLE_HTML)
        self.assertEqual(day, 'Mon')

    def test_parse_date_str(self):
        _, date_str, _, _ = parse_draw_page(self.SAMPLE_HTML)
        self.assertEqual(date_str, '08 Apr 2024')

    def test_parse_all_six_winning_numbers(self):
        _, _, numbers, _ = parse_draw_page(self.SAMPLE_HTML)
        self.assertEqual(numbers, ['12', '23', '24', '34', '43', '46'])

    def test_parse_additional_number(self):
        _, _, _, additional = parse_draw_page(self.SAMPLE_HTML)
        self.assertEqual(additional, '42')

    def test_missing_draw_date_returns_none(self):
        self.assertIsNone(parse_draw_page('<html><body></body></html>'))

    def test_missing_additional_returns_none(self):
        html = "<th class='drawDate'>Mon, 01 Jan 2024</th>"
        _, _, _, additional = parse_draw_page(html)
        self.assertIsNone(additional)


class TestParseLocations(unittest.TestCase):
    def _wrap(self, group1_li, group2='<p>Group 2 winning tickets sold at:</p>'):
        return '<ul>{}</ul>{}'.format(group1_li, group2)

    def test_no_winner_returns_empty(self):
        self.assertEqual(parse_locations('Group 1 has no winner this draw'), [])

    def test_non_quickpick_winner(self):
        html = self._wrap('<li>Nan Huat Wine Store - Blk 513 Bishan St 13 #01-508 ( 1 System 7 Entry )</li>')
        results = parse_locations(html)
        self.assertEqual(len(results), 1)
        _, place, address, draw, system = results[0]
        self.assertEqual(place.strip(), 'Nan Huat Wine Store')
        self.assertEqual(draw, 'No QuickPick')
        self.assertEqual(system, 'System 7')

    def test_quickpick_winner(self):
        html = self._wrap('<li>NTUC FP Hougang Mall - 90 Hougang Ave 10 #B1-07 ( 1 QuickPick System 7 Entry )</li>')
        results = parse_locations(html)
        self.assertEqual(len(results), 1)
        _, place, address, draw, system = results[0]
        self.assertEqual(draw, 'QuickPick')
        self.assertEqual(system, 'System 7')

    def test_malformed_entry_is_skipped(self):
        html = self._wrap('<li>no dash or parentheses here</li>')
        self.assertEqual(parse_locations(html), [])

    def test_multiple_winners(self):
        html = self._wrap(
            '<li>Store A - Addr A ( 1 System 7 Entry )</li>'
            '<li>Store B - Addr B ( 1 System 12 Entry )</li>'
        )
        results = parse_locations(html)
        self.assertEqual(len(results), 2)


class TestFetchDraw(unittest.TestCase):
    def test_encodes_draw_number_correctly(self):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = '<html></html>'
        mock_session.get.return_value = mock_resp

        fetch_draw(mock_session, '3963')

        _, call_kwargs = mock_session.get.call_args
        sppl = call_kwargs['params']['sppl']
        decoded = base64.b64decode(sppl).decode()
        self.assertEqual(decoded, 'DrawNumber=3963')

    def test_raises_on_http_error(self):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError('404')
        mock_session.get.return_value = mock_resp

        with self.assertRaises(requests.HTTPError):
            fetch_draw(mock_session, '3963')


class TestParsePrizeTable(unittest.TestCase):
    SAMPLE_HTML = """
    <table class='tableWinningShares'>
      <thead><tr><th>Prize Group</th><th>Share Amount</th><th>No. of Winning Shares</th></tr></thead>
      <tbody>
        <tr><td>Group 1</td><td>$3,044,451</td><td>1</td></tr>
        <tr><td>Group 2</td><td>$127,495</td><td>3</td></tr>
        <tr><td>Group 7</td><td>$10</td><td>135,895</td></tr>
      </tbody>
    </table>
    """

    def test_extracts_all_rows(self):
        results = parse_prize_table(self.SAMPLE_HTML)
        self.assertEqual(len(results), 3)

    def test_prize_group_numbers(self):
        results = parse_prize_table(self.SAMPLE_HTML)
        groups = [r[0] for r in results]
        self.assertEqual(groups, [1, 2, 7])

    def test_prize_amount_in_cents(self):
        results = parse_prize_table(self.SAMPLE_HTML)
        self.assertEqual(results[0][1], 304445100)  # $3,044,451 * 100

    def test_winner_count(self):
        results = parse_prize_table(self.SAMPLE_HTML)
        self.assertEqual(results[1][2], 3)

    def test_missing_table_returns_empty(self):
        self.assertEqual(parse_prize_table('<html></html>'), [])


class TestParseAllWinnerLocations(unittest.TestCase):
    SAMPLE_HTML = (
        '<p>Group 1 winning tickets sold at:</p>'
        '<ul><li>Store A - Addr A ( 1 System 7 Entry )</li></ul>'
        '<p>Group 2 winning tickets sold at:</p>'
        '<ul><li>Store B - Addr B ( 1 QuickPick System 7 Entry )</li>'
        '<li>Store C - Addr C ( 1 System 8 Entry )</li></ul>'
    )

    def test_extracts_both_groups(self):
        results = parse_all_winner_locations(self.SAMPLE_HTML)
        groups = [r[0] for r in results]
        self.assertIn(1, groups)
        self.assertIn(2, groups)

    def test_total_entries(self):
        results = parse_all_winner_locations(self.SAMPLE_HTML)
        self.assertEqual(len(results), 3)

    def test_group1_non_quickpick(self):
        results = parse_all_winner_locations(self.SAMPLE_HTML)
        g1 = [r for r in results if r[0] == 1]
        self.assertEqual(len(g1), 1)
        self.assertEqual(g1[0][4].strip(), 'No QuickPick')

    def test_group2_quickpick(self):
        results = parse_all_winner_locations(self.SAMPLE_HTML)
        g2_qp = [r for r in results if r[0] == 2 and r[4] == 'QuickPick']
        self.assertEqual(len(g2_qp), 1)

    def test_no_groups_returns_empty(self):
        self.assertEqual(parse_all_winner_locations('<html></html>'), [])


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        sys.argv.pop(1)
        unittest.main()
    else:
        main()

