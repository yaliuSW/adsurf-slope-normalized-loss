"""Meaningful boundary checks for the observable-only switch controller."""
import unittest
import numpy as np
from snadsurf.switching import PlateauSwitch


class TestPlateauSwitch(unittest.TestCase):
    def test_independent_starts_and_scale_invariance(self):
        a=PlateauSwitch(3,window=2,tolerance=.01,patience=2,min_raw=4)
        for k in range(9):
            a.observe(k,[1.,1e6,2.**(-k)],(0,))
        self.assertEqual(a.switch_updates.tolist(),[4,4,-1])
        self.assertEqual(len(a.events),2)

    def test_initial_burnin_and_one_way_switch(self):
        a=PlateauSwitch(1,window=2,patience=2,min_raw=8)
        for k in range(15):
            a.observe(k,[1. if k<9 else 2.**(-k)],(0,))
        self.assertEqual(a.switch_updates.tolist(),[8])
        self.assertEqual(len(a.events),1)

    def test_new_data_resets_incomparable_loss_history(self):
        a=PlateauSwitch(1,window=2,patience=2,min_raw=0)
        for k in range(9):
            a.observe(k,[10. if k<4 else 1.],(0,) if k<4 else (0,1),k>=4)
        self.assertEqual(a.switch_updates.tolist(),[8])
        self.assertEqual(a.events[0]['regime'],[0,1])

    def test_no_switch_without_permission_or_at_budget(self):
        a=PlateauSwitch(1,window=2,patience=2,min_raw=4)
        for k in range(5):
            a.observe(k,[0.],(0,),allow_switch=k<4)
        self.assertEqual(a.switch_updates.tolist(),[-1])

    def test_nonfinite_and_skipped_updates_are_rejected(self):
        a=PlateauSwitch(1)
        with self.assertRaises(ValueError):a.observe(0,[np.nan],(0,))
        a.observe(0,[1.],(0,))
        with self.assertRaises(ValueError):a.observe(2,[1.],(0,))


if __name__=='__main__':unittest.main()
