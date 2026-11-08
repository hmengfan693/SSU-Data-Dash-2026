"""Boundary and pooling tests for the official monthly validation adapter."""
import ast
import sys
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_official import split_month, score, official_score
import data


class MonthlyValidationTests(unittest.TestCase):
    def test_cutoff_is_exclusive_and_forecast_end_is_exclusive(self):
        raw = pd.DataFrame(dict(ml_num=['c', 'a', 'b', 'd'], date_sold=pd.to_datetime(
            ['2026-01-01', '2025-11-30', '2025-12-01', '2025-12-31'])))
        train, valid, cutoff, end = split_month(raw, '2025-12')
        self.assertEqual(train.ml_num.tolist(), ['a'])
        self.assertEqual(valid.ml_num.tolist(), ['b', 'd'])
        self.assertEqual((cutoff, end), ('2025-12-01', '2026-01-01'))

    def test_pooled_median_is_not_average_of_group_medians(self):
        rows = pd.DataFrame(dict(sold_price=[100., 100., 100., 100.],
                                 predicted_price=[100., 101., 102., 200.], month=['a', 'a', 'a', 'b']))
        self.assertEqual(score(rows)['mdape'], 1.5)
        self.assertNotEqual(score(rows)['mdape'], np.mean([score(g)['mdape'] for _, g in rows.groupby('month')]))
        self.assertEqual(official_score(rows)['mdape'], 1.5)

    def test_query_columns_match_track_availability(self):
        rows = pd.DataFrame({c: [1.] for c in [*data.AFTER_SALE_COLUMNS, 'list_price', 'bedrooms']})
        self.assertEqual(list(data.hide_answers(rows, 'off_market')), ['bedrooms'])
        self.assertEqual(list(data.hide_answers(rows, 'list_aware')), ['list_price', 'bedrooms'])

    def test_duplicate_ids_are_rejected(self):
        rows = pd.DataFrame(dict(ml_num=['a', 'a'], date_sold=pd.to_datetime(['2025-11-30', '2025-12-01'])))
        with self.assertRaises(ValueError):
            split_month(rows, '2025-12')

    def test_all_score_fields_match_official_source(self):
        source = Path(__file__).resolve().parents[1] / 'docs' / 'official' / 'validation_2026-11-08' / 'run.py'
        tree = ast.parse(source.read_text())
        functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ['bootstrap_ci', 'metrics']], type_ignores=[])
        namespace = {'np': np}
        exec(compile(functions, str(source), 'exec'), namespace)
        rows = pd.DataFrame(dict(sold_price=[121., 997., 999., 1800., 11000.],
                                 predicted_price=[100., 1010., 950., 1850., 10000.]))
        self.assertEqual(official_score(rows), namespace['metrics'](rows))


if __name__ == '__main__':
    unittest.main()
