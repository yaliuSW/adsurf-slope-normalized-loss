"""Selection must reject incomplete or wrong branches without using truth."""
import unittest
import numpy as np
from snadsurf.selection import score_spectrum


class TestSelection(unittest.TestCase):
    def test_known_order_is_enforced(self):
        obs=np.array([[1.,.2,0],[2.,.3,0]])
        sp=np.array([[.4,.2],[.5,.3]])
        row,_,_=score_spectrum(sp,obs,np.arange(2))
        self.assertEqual(row['status'],'wrong_branch')

    def test_incomplete_predictions_are_not_ranked(self):
        obs=np.array([[1.,.2,0],[2.,.3,0]])
        row,_,_=score_spectrum(np.array([[.2,.5],[np.nan,.6]]),obs,np.arange(2))
        self.assertEqual(row['status'],'missing_root')
        self.assertNotIn('weighted_RMSE_m_s',row)

    def test_full_branch_tie_is_ambiguous(self):
        obs=np.array([[1.,.2,0],[2.,.3,0]])
        row,_,_=score_spectrum(np.array([[.2,.2],[.3,.3]]),obs,np.arange(2))
        self.assertEqual(row['status'],'ambiguous_branch')

    def test_unknown_higher_groups_get_consistent_ordered_branches(self):
        sp=np.array([[.2,.35,.5,.7],[.3,.4,.6,.8]])
        obs=np.array([[1.,.2,0],[2.,.3,0],[1.,.5,1],[2.,.6,1],[1.,.7,2],[2.,.8,2]])
        row,pred,modes=score_spectrum(sp,obs,np.array([0,1,0,1,0,1]),field=True)
        self.assertEqual(row['status'],'accepted')
        self.assertEqual(modes.tolist(),[0,0,2,2,3,3])
        np.testing.assert_array_equal(pred,obs[:,1])


if __name__=='__main__':unittest.main()
