"""Diagnostic partial-root scores must preserve group weighting and missingness."""
import unittest
import numpy as np
from examples.evaluate_history import available_point_score


class TestHistory(unittest.TestCase):
    def test_partial_group_retains_four_to_one_weight(self):
        obs=np.array([[1.,.2,0],[2.,.3,0],[1.,.5,1]])
        score,counts=available_point_score(obs,np.array([.203,np.nan,.504]))
        self.assertAlmostEqual(score,np.sqrt((4*9+16)/5))
        self.assertEqual(counts,{'group0_points':1,'group1_points':1})

    def test_missing_entire_group_is_not_silently_dropped(self):
        obs=np.array([[1.,.2,0],[1.,.5,1]])
        score,_=available_point_score(obs,np.array([.203,np.nan]))
        self.assertTrue(np.isnan(score))


if __name__=='__main__':unittest.main()
