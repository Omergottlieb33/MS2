"""Tracking regressions, runnable with unittest (no pytest required)."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from cell_tracking import cell_tracking, evaluate_tracklets
from src.track import create_tracklets, get_border_labels, optional_assignment
from src.track import match_cells_by_iou_hungarian_local_optimized as match
from src.track_unify import stitch_gaps

class TrackingTests(unittest.TestCase):
    def test_overlap_and_scaled_z(self):
        a = np.zeros((12,30,30),np.uint16); b = a.copy()
        a[1:3,10:15,10:15]=1
        b[1:3,10:15,14:19]=2
        b[8:10,10:15,10:15]=3
        for xy in (True, False):
            self.assertEqual(match(a,b,max_centroid_distance=15,use_2d_distance=xy,dist_weight=0 if xy else .3,unmatched_cost=1 if xy else .85),{2:1})
    def test_exact_iou(self):
        a=np.zeros((2,80,80),np.uint16); a[:,10:70,10:70]=1
        self.assertEqual(match(a,a,search_radius=1,min_iou=.99),{1:1})
    def test_optional_assignment(self):
        self.assertEqual(optional_assignment([[.9]],.85),[])
        self.assertEqual(optional_assignment([[.1,.9],[.2,np.inf]],.85),[(0,0)])
        self.assertEqual(optional_assignment(np.empty((0,3))),[])
    def test_first_frame_and_border_recovery(self):
        for y in (0,10):
            a=np.zeros((3,30,30),np.uint16); a[:,y:y+4,10:14]=1
            tracks=create_tracklets([{},{}],skip_matches=[],masks=[a,np.zeros_like(a),a*3])
            self.assertEqual(list(tracks.values()),[[1,-1,3]])
    def test_singletons_and_interior_gaps(self):
        masks=[np.zeros((2,40,40),np.uint16) for _ in range(3)]
        for i in range(3): masks[i][:,15:18,15:18]=i+1
        rows=list(create_tracklets([{},{}],masks=masks).values())
        self.assertEqual(rows,[[1,-1,-1],[-1,2,-1],[-1,-1,3]])
        self.assertEqual(list(create_tracklets([],masks=[masks[0]]).values()),[[1]])
    def test_no_masks_skip_compatibility(self):
        self.assertEqual(list(create_tracklets([{},{}],skip_matches=[{3:1}]).values()),[[1,-1,3]])
    def test_border_contact(self):
        a=np.zeros((2,40,40),np.uint16); a[:,:20,10:20]=1
        self.assertEqual(get_border_labels(a),{1})
    def test_empty(self):
        a=np.zeros((2,20,20),np.uint16)
        self.assertEqual(create_tracklets([{}],masks=[a,a]),{})
        self.assertTrue(evaluate_tracklets({}).empty)
    def test_stitch_chains(self):
        for order in ((0,1,2),(2,0,1)):
            rows={0:[1,-1,-1,-1,-1,-1],1:[-1,-1,2,-1,-1,-1],2:[-1,-1,-1,-1,3,-1]}
            props={i:{} for i in range(6)}
            for t,label in ((0,1),(2,2),(4,3)): props[t][label]=(5,20,20,1200)
            result=stitch_gaps({k:rows[k] for k in order},props,list(range(6)),{i:1200 for i in range(6)},[])
            self.assertEqual(list(result.values()),[[1,-1,2,-1,3,-1]])
    def test_frame_limit_output_and_missing_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); masks=root/'masks'; masks.mkdir()
            for t in (0,2):
                np.savez(masks/f'z_stack_t{t}_seg_masks.npz',masks=np.ones((2,4,4),np.uint16))
            path=root/'result.json'
            self.assertEqual(list(cell_tracking(str(masks),t=1,output_path=str(path)).values()),[[1]])
            self.assertEqual(json.loads(path.read_text()),{'0':[1]})
            with self.assertRaises(ValueError): cell_tracking(str(masks),t=None)
if __name__=='__main__':
    unittest.main()
