# get all locations where one can buy TOTO
import sqlite3, re, logging, unittest
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

OUTLETS_URL = 'https://www.singaporepools.com.sg/outlets/Pages/lo_results.aspx?sppl=cz0mej1BJm89QSZjPUEmZD1B'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36'
}


def parse_outlet(item_text):
    """Parse the text of a single outlet <li> element.

    Returns (location, address, postal_code) or None if the entry is malformed
    or has no recognisable Singapore postal code.
    """
    parts = item_text.replace('\xa0', '').replace('  ', ' ').split('\n')
    if len(parts) < 3:
        return None
    location = parts[1].strip()
    address = parts[2].strip()
    match = re.search(r'Singapore\s\d+', address)
    if match is None:
        return None
    postal_code = match.group()
    return location, address, postal_code


def main():
    conn = sqlite3.connect('toto.sqlite')
    try:
        cur = conn.cursor()

        resp = requests.get(OUTLETS_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        count = 0
        for item in soup.select('table[id="tblOutletSearchResult"] li'):
            try:
                result = parse_outlet(item.getText())
                if result is None:
                    logging.warning('Skipping outlet: could not parse "%s"', item.getText()[:80])
                    continue
                location, address, postal_code = result
                cur.execute(
                    "INSERT OR IGNORE INTO placeall(location,address,postal_code) VALUES(?,?,?)",
                    (location, address, postal_code),
                )
                count += 1
            except Exception as e:
                logging.warning('Skipping malformed outlet entry: %s', e)

        conn.commit()
        logging.info('Inserted %d outlet locations.', count)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestParseOutlet(unittest.TestCase):
    def _make(self, location, address):
        return '\n{}\n{}\n'.format(location, address)

    def test_valid_entry(self):
        text = self._make('Singapore Pools Bishan Branch', 'Blk 513 Bishan St 13 #01-508 Singapore 570513')
        result = parse_outlet(text)
        self.assertIsNotNone(result)
        location, address, postal_code = result
        self.assertEqual(location, 'Singapore Pools Bishan Branch')
        self.assertEqual(address, 'Blk 513 Bishan St 13 #01-508 Singapore 570513')
        self.assertEqual(postal_code, 'Singapore 570513')

    def test_nbsp_replaced(self):
        text = '\n\xa0Store Name\xa0\nBlk 1 Addr Singapore 123456\n'
        result = parse_outlet(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 'Store Name')

    def test_no_postal_code_returns_none(self):
        text = self._make('Some Store', 'Blk 1 Some Street #01-01')
        self.assertIsNone(parse_outlet(text))

    def test_too_few_parts_returns_none(self):
        self.assertIsNone(parse_outlet('only one line'))

    def test_strips_whitespace(self):
        text = '\n  My Store  \n  123 Road Singapore 654321  \n'
        result = parse_outlet(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 'My Store')
        self.assertEqual(result[1], '123 Road Singapore 654321')

    def test_postal_code_extracted_correctly(self):
        text = self._make('Store', '10 Bayfront Ave Singapore 018956')
        _, _, postal_code = parse_outlet(text)
        self.assertEqual(postal_code, 'Singapore 018956')


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        sys.argv.pop(1)
        unittest.main()
    else:
        main()


