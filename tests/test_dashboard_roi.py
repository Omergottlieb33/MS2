"""ROI validation and physical-to-pixel centroid regression tests."""
import unittest
import numpy as np
from src.tracking_dashboard import validate_roi
from src.tracklet_quality import diagnostics, settings_checked

class RoiTests(unittest.TestCase):
    def setUp(self):
        self.report={'frames':3,'paths':{'tracklets':'/recording/unify/tracks.json'}}
        self.roi={'roi_first':[[0,0],[4,0],[4,4]],'t_first':0}
    def test_valid_single_endpoint(self):
        validate_roi(self.roi,self.report)
        validate_roi({'roi_last':self.roi['roi_first'],'t_last':-1},self.report)
    def test_invalid_geometry_and_time(self):
        for changes in ({'roi_first':[]},{'roi_first':[[0,0]]},
                        {'t_first':3},{'t_first':.5},
                        {'roi_first':[[0,0],[1,0],[float('nan'),1]]}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                validate_roi({**self.roi,**changes},self.report)
    def test_wrong_recording(self):
        with self.assertRaises(ValueError):
            validate_roi({**self.roi,'tracklets':'/another/unify/tracks.json'},self.report)
    def test_pixel_centroids_and_missing_frame(self):
        s=settings_checked({'voxel_size_um':[2,3,4]})
        result=diagnostics({'a':np.array([1,-1])},
                           [{1:{'centre':[2,6,12],'volume':24}},{}],
                           [{1:'a'},{}],s)[0]
        self.assertEqual(result['centers_xy'],[[3,2],None])

if __name__=='__main__':
    unittest.main()
