"""Ground-truth-free metrics: explicit populations, denominators and exclusions."""
import copy
import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
import numpy as np
from src.tracking_evaluation import (DEFAULTS, association_edges, boundary, features,
    fraction, load_rows, map_labels, score_dataset, stability, stress_suite)
from src.tracking_dashboard import handler_for

class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.s={**DEFAULTS,'min_volume':1,'xy_margin':1,'z_margin':1,
                'voxel_size_um':[1,1,1],'motion_threshold_um_per_frame':.5}
    def feature(self, x=5, volume=10):
        return dict(centre=[5,5,x],volume=volume,box=[[4,7],[4,7],[4,7]])
    def score(self, rows, props, refs=None, maps=None):
        n=len(props); owners=[{} for _ in props]
        for tid,row in rows.items():
            for t,label in enumerate(row):
                if label>0: owners[t][label]=tid
        maps=maps or [{l:l for l in p} for p in props]
        return score_dataset('test',{k:np.array(v) for k,v in rows.items()},owners,props,maps,refs or props,(12,12,12),self.s)
    def test_undefined_is_not_zero_or_perfect(self):
        self.assertIsNone(fraction(0,0))
        self.assertIsNone(stability(set(),set())['jaccard'])
    def test_boundary_separates_axial_contact(self):
        f=self.feature(); f['box'][0]=[0,7]
        self.assertEqual(boundary(f,(12,12,12),self.s),'axial_uncertain')
        f['box'][1]=[0,7]
        self.assertEqual(boundary(f,(12,12,12),self.s),'xy')
    def test_fixed_coverage_and_cohort(self):
        props=[{1:self.feature(),2:self.feature(8)} for _ in range(3)]
        r=self.score({'a':[1,1,1]},props)
        self.assertEqual(r['coverage'],.5)
        self.assertEqual(r['eligible_observations'],6)
        self.assertEqual(r['cohort_size'],2)
        self.assertEqual(r['cohort_curve'][2]['value'],.5)
    def test_final_frame_not_an_ending(self):
        props=[{1:self.feature()} for _ in range(3)]
        r=self.score({'a':[1,1,1]},props)
        self.assertEqual(r['endings']['interior'],0)
        self.assertEqual(r['exposure']['interior'],2)
    def test_normalized_ending_and_gap_continuity(self):
        props=[{1:self.feature()} for _ in range(4)]
        r=self.score({'a':[1,-1,1,-1]},props)
        self.assertEqual(r['interior_endings_per_1000'],500)
        self.assertEqual(r['cohort_curve'][2]['value'],1)
        self.assertEqual(r['cohort_curve'][2]['uninterrupted'],0)
    def test_motion_uses_elapsed_frames(self):
        props=[{1:self.feature(x=t+4)} for t in range(5)]
        r=self.score({'a':[1,-1,1,-1,1]},props)
        self.assertEqual(r['motion_tests'],1)
        self.assertEqual(r['motion_flags'],0)
        props[-1][1]['centre'][2]=11
        r=self.score({'a':[1,-1,1,-1,1]},props)
        self.assertEqual(r['motion_flags_per_1000'],1000)
    def test_reference_volume_controls_eligibility(self):
        self.s['min_volume']=100
        refs=[{1:self.feature(volume=200)}]*3
        props=[{7:self.feature(volume=10)}]*3
        r=self.score({'a':[7,7,7]},props,refs,[{7:1}]*3)
        self.assertEqual(r['coverage'],1)
        self.assertEqual(r['cohort_size'],1)
    def test_spatial_mapping_and_splits(self):
        a=np.zeros((4,8,8),dtype=np.uint16); a[:,2:6,2:6]=1
        b=a*7
        mapping,counts=map_labels(a,b,.25)
        self.assertEqual(mapping,{7:1})
        b[:2][b[:2]>0]=8
        mapping,counts=map_labels(a,b,.25)
        self.assertEqual(len(mapping),1)
        self.assertEqual(counts['split_like'],1)
        self.assertEqual(counts['unmapped_output'],1)
    def test_duplicates_and_length_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'tracks.json';p.write_text(json.dumps({'a':[1,1],'b':[1,2]}))
            with self.assertRaises(ValueError): load_rows(p,2)
            p.write_text(json.dumps({'a':[1]}))
            with self.assertRaises(ValueError): load_rows(p,2)
    def test_association_ids_do_not_matter(self):
        a=association_edges({'0':[1,-1,2]})
        b=association_edges({'999':[1,-1,2]})
        self.assertEqual(stability(a,b)['retention'],1)
    def test_stress_recovery_and_retention(self):
        masks=[]
        for t in range(5):
            m=np.zeros((3,20,20),np.uint16);m[:,8:12,8:12]=1;masks.append(m)
        opts=dict(start=0,frames=5,drop_fraction=1,max_removals=1,seed=0)
        r=stress_suite(masks,{'a':np.array([1]*5)},self.s,opts,progress=lambda _:None)
        self.assertEqual(r['variants']['reverse_time']['jaccard'],1)
        self.assertEqual(r['variants']['dropout']['recovery'],1)
        self.assertEqual(r['variants']['dropout']['detection_retention'],1)
        self.assertEqual(r['variants']['dropout']['jaccard'],1)
    def test_http_and_standalone_export(self):
        report=dict(datasets=[],title='x</script>',settings={},notes=[],frames=1,stress=None)
        server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(report))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            url=f'http://127.0.0.1:{server.server_port}'
            self.assertEqual(json.load(urllib.request.urlopen(url+'/api/report'))['title'],'x</script>')
            html=urllib.request.urlopen(url).read().decode()
            self.assertIn('report-data',html)
            self.assertNotIn('x</script>',html)
            self.assertEqual(json.load(urllib.request.urlopen(url+'/health'))['status'],'ok')
        finally:
            server.shutdown();server.server_close();thread.join()
if __name__=='__main__':
    unittest.main()
