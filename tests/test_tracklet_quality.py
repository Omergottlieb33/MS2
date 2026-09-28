"""Checks for the four single-track dashboard diagnostics."""
import unittest
import numpy as np
from src.tracklet_quality import diagnostics, frame_properties, neighborhoods, settings_checked

class QualityTests(unittest.TestCase):
    def score(self, rows, props):
        rows={k:np.array(v) for k,v in rows.items()}
        owners=[{int(row[t]):tid for tid,row in rows.items() if row[t]>0} for t in range(len(props))]
        return diagnostics(rows,props,owners,settings_checked({'voxel_size_um':[1,1,1]}))
    def prop(self,x=0,v=1):
        return dict(centre=[0,0,x],volume=v)
    def test_recording_denominator_and_gaps(self):
        r=self.score({'a':[1,-1,1,-1]},[{1:self.prop()}]*4)[0]
        self.assertEqual(r['length_fraction'],.5)
        self.assertEqual(r['gap_frames'],1)
        self.assertEqual(r['step_tests'],0)
        self.assertIsNone(r['step_p95'])
    def test_singletons_are_unmeasurable(self):
        r=self.score({'a':[1]},[{1:self.prop()}])[0]
        for key in ('volume_cv','step_p95','neighbor_retention'):
            self.assertIsNone(r[key])
    def test_volume_cv_and_motion(self):
        r=self.score({'a':[1,2]},[{1:self.prop(0,1)},{2:self.prop(4,3)}])[0]
        self.assertEqual(r['volume_cv'],.5)
        self.assertEqual(r['step_p95'],4)
        self.assertEqual(r['large_steps'],1)
    def test_physical_geometry(self):
        mask=np.zeros((3,4,5),dtype=np.uint16)
        mask[1,1:3,2:4]=7
        p=frame_properties(mask,np.array([2,3,4]))[7]
        self.assertEqual(p['volume'],96)
        np.testing.assert_allclose(p['centre'],[2,4.5,10])
    def test_neighbor_denominator_includes_untracked(self):
        props=[{1:self.prop(),2:self.prop(1),3:self.prop(2)}]*2
        r=self.score({'a':[1,1],'b':[2,2]},props)[0]
        self.assertEqual(r['neighbor_retention'],.5)
        self.assertEqual(r['neighbor_total'],2)
    def test_neighbor_identity_break(self):
        props=[{1:self.prop(),2:self.prop(1)}]*2
        r=self.score({'a':[1,1],'b':[2,-1],'c':[-1,2]},props)[0]
        self.assertEqual(r['neighbor_retention'],0)
    def test_neighbor_radius_and_limit(self):
        props={1:self.prop(),2:self.prop(1),3:self.prop(2),4:self.prop(20)}
        n=neighborhoods(props,{1:'a',2:'b',3:'c',4:'d'},10,1)
        self.assertEqual(n[1],dict(count=1,tracks={'b'}))
        self.assertEqual(n[4]['count'],0)
    def test_invalid_settings(self):
        for s in ({'neighbor_k':0},{'max_step_um':float('nan')},{'voxel_size_um':[1,0,1]}):
            with self.assertRaises(ValueError): settings_checked(s)

if __name__=='__main__':
    unittest.main()
