import sqlite3, json, logging, os, unittest
import requests
from unittest.mock import MagicMock, patch

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

GEOCODE_URL = 'https://maps.googleapis.com/maps/api/geocode/json'


def parse_geocode_response(data):
    """Extract (lat, lng) from a Google Geocode API response dict.

    Returns (lat, lng) on success, or None if the response has no results.
    """
    if not data.get('results'):
        return None
    location = data['results'][0]['geometry']['location']
    return location['lat'], location['lng']


def geocode(session, postal_code, api_key):
    """Call the Geocode API for a postal code string.

    Returns (lat, lng) or None if no result.
    Raises requests.HTTPError on a bad HTTP status.
    """
    resp = session.get(
        GEOCODE_URL,
        params={'address': postal_code, 'key': api_key},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    result = parse_geocode_response(data)
    if result is None:
        logging.warning('No results for postal_code=%s (status=%s)', postal_code, data.get('status'))
    return result


def main():
    # API key must be set in environment variable GOOGLE_API_KEY
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise EnvironmentError('GOOGLE_API_KEY environment variable is not set.')

    conn = sqlite3.connect('toto.sqlite')
    try:
        cur = conn.cursor()
        session = requests.Session()
        query = cur.execute('SELECT postal_code FROM placeall WHERE scanned=0').fetchall()

        for row in query:
            for postal_code in row:
                logging.info('Geocoding: %s', postal_code)
                try:
                    result = geocode(session, postal_code, api_key)
                    if result is None:
                        continue
                    lat, lng = result
                    cur.execute(
                        'UPDATE placeall SET latitude=?,longitude=?,scanned=? WHERE postal_code=?',
                        (lat, lng, 1, postal_code),
                    )
                except Exception as e:
                    logging.warning('Failed to geocode %s: %s', postal_code, e)

        conn.commit()
        logging.info('Done.')
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestParseGeocodeResponse(unittest.TestCase):
    def _make_response(self, lat, lng):
        return {
            'status': 'OK',
            'results': [{'geometry': {'location': {'lat': lat, 'lng': lng}}}]
        }

    def test_valid_response_returns_lat_lng(self):
        data = self._make_response(1.3521, 103.8198)
        self.assertEqual(parse_geocode_response(data), (1.3521, 103.8198))

    def test_empty_results_returns_none(self):
        self.assertIsNone(parse_geocode_response({'status': 'ZERO_RESULTS', 'results': []}))

    def test_missing_results_key_returns_none(self):
        self.assertIsNone(parse_geocode_response({'status': 'REQUEST_DENIED'}))

    def test_negative_coordinates(self):
        data = self._make_response(-33.8688, 151.2093)
        lat, lng = parse_geocode_response(data)
        self.assertAlmostEqual(lat, -33.8688)
        self.assertAlmostEqual(lng, 151.2093)


class TestGeocode(unittest.TestCase):
    def _mock_session(self, response_data, http_error=None):
        session = MagicMock()
        resp = MagicMock()
        if http_error:
            resp.raise_for_status.side_effect = http_error
        else:
            resp.json.return_value = response_data
        session.get.return_value = resp
        return session

    def test_returns_lat_lng_on_success(self):
        data = {'status': 'OK', 'results': [{'geometry': {'location': {'lat': 1.3, 'lng': 103.8}}}]}
        session = self._mock_session(data)
        result = geocode(session, 'Singapore 570513', 'fake_key')
        self.assertEqual(result, (1.3, 103.8))

    def test_passes_postal_code_and_key_as_params(self):
        data = {'status': 'OK', 'results': [{'geometry': {'location': {'lat': 1.0, 'lng': 100.0}}}]}
        session = self._mock_session(data)
        geocode(session, 'Singapore 123456', 'mykey')
        _, call_kwargs = session.get.call_args
        self.assertEqual(call_kwargs['params']['address'], 'Singapore 123456')
        self.assertEqual(call_kwargs['params']['key'], 'mykey')

    def test_returns_none_on_zero_results(self):
        data = {'status': 'ZERO_RESULTS', 'results': []}
        session = self._mock_session(data)
        self.assertIsNone(geocode(session, 'Singapore 999999', 'fake_key'))

    def test_raises_on_http_error(self):
        session = self._mock_session({}, http_error=requests.HTTPError('403'))
        with self.assertRaises(requests.HTTPError):
            geocode(session, 'Singapore 123456', 'fake_key')

    def test_timeout_is_set(self):
        data = {'status': 'OK', 'results': [{'geometry': {'location': {'lat': 1.0, 'lng': 100.0}}}]}
        session = self._mock_session(data)
        geocode(session, 'Singapore 123456', 'fake_key')
        _, call_kwargs = session.get.call_args
        self.assertEqual(call_kwargs['timeout'], 10)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        sys.argv.pop(1)
        unittest.main()
    else:
        main()

