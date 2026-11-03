import unittest
import pandas as pd
from enhanced import Features


class FeatureTests(unittest.TestCase):
    def test_physical_units_and_order_encoding(self):
        frame=pd.DataFrame(dict(current_A=[200],voltage_V=[25],ambient_C=[80],speed_m_s=[.005],process_order=[2]))
        out=Features(physical=True,pooled=True).fit_transform(frame)
        self.assertEqual(out.shape,(1,7))
        self.assertEqual(out[0,4],5000)
        self.assertEqual(out[0,5],1000000)
        self.assertEqual(out[0,6],1)
