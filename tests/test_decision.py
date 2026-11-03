import unittest
from unittest.mock import patch
import decision


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.bundle={'features':['current_A'],'bounds':{'current_A':[180,240]}}
        self.request={'current':{'current_A':210},'steps':{'current_A':5},'max_change':{'current_A':6},'orders':[1,2]}

    @patch('decision.w.predict')
    def test_lattice_respects_change_and_enumerates_orders(self,mock):
        frame=decision.grid(self.bundle,self.request)
        self.assertEqual(set(frame.current_A),{205,210,215})
        self.assertEqual(len(frame),6)

    @patch('decision.w.predict')
    def test_invalid_locked_step_rejected(self,mock):
        self.request['locked']={'current_A':211}
        with self.assertRaises(ValueError):
            decision.grid(self.bundle,self.request)

    @patch('decision.w.predict')
    def test_negative_step_rejected(self,mock):
        self.request['steps']['current_A']=-1
        with self.assertRaises(ValueError):
            decision.grid(self.bundle,self.request)
