import unittest
import numpy as np
import welding


class ValidationTests(unittest.TestCase):
    def test_imported_physical_outputs(self):
        frame = welding.read_success(welding.ROOT / 'clean_data_1219.csv')
        self.assertEqual(len(frame), 1219)
        self.assertTrue((frame.max_displacement_m + 1e-12 >= frame[['max_ux_m','max_uy_m','max_uz_m']].max(axis=1)).all())

    def test_group_partitions_and_provenance(self):
        directory = welding.prepare()
        frame = welding.pd.read_csv(directory / 'dataset.csv')
        self.assertEqual(frame.groupby('group_id').split.nunique().max(), 1)
        self.assertEqual(set(frame.split), {'train', 'calibration', 'test'})
        self.assertFalse(frame.duplicated(welding.INPUTS).any())
        self.assertTrue((directory / 'source_records.csv').exists())

    def test_reject_extrapolation(self):
        bundle = {'bounds': {'current_A': [180,240]}}
        values = dict(zip(welding.INPUTS, [300,24,20,.004,1]))
        with self.assertRaises(ValueError):
            welding.predict(bundle, values)

    def test_reject_fractional_order(self):
        values = dict(zip(welding.INPUTS, [210,24,20,.004,1.5]))
        with self.assertRaises(ValueError):
            welding.predict({}, values)


if __name__ == '__main__':
    unittest.main()
