"""ms2/app/analysis.py: the analysis page finds every result in both folder layouts, with its
tracks, masks and image, and gives each stage window's cells as the comparison measures them.

    python tests/test_analysis.py
"""
import json
import os
import shutil
import tempfile

import numpy as np
import tifffile

from fixtures import passed, write_recording

from ms2.app.analysis import discover, find_matrices, sample_data
from ms2.cell_activity_compare import cell_positions
from ms2.gene_expression.expression_matrix import RESULTS_NAME
from ms2.track_diagnostics import cell_properties

SIGNALS = np.array([[0, 30, 40, 0, 0, 0],
                    [0, 0, np.nan, 0, 50, 0],           # untracked on frame 2
                    [np.nan, np.nan, 0, 0, 0, 0],       # born on frame 2
                    [0, 0, 0, 0, 0, 0]], dtype=float)


def pipeline_result(root, signals=SIGNALS, name='rec-a'):
    """<root>/STAGE-12/<name>, laid out as ms2/pipeline.py writes it."""
    rec = os.path.join(root, 'STAGE-12', name)
    r = write_recording(os.path.join(rec, 'gene_expression'), signals, name=RESULTS_NAME)
    os.makedirs(os.path.join(rec, 'unified'))
    shutil.move(r['tracklets'], os.path.join(rec, 'unified', 'tracklets_unified.json'))
    shutil.move(r['masks_dir'], os.path.join(rec, 'unified', 'masks_split'))
    os.makedirs(os.path.join(rec, 'preprocess'))
    tifffile.imwrite(os.path.join(rec, 'preprocess', f'SUM_C2-{name}.tif'),
                     np.zeros((np.asarray(signals).shape[1], 16, 16), np.float32))
    return rec


def split_result(root, tracks_rec):
    """A hand-run result in a folder of its own whose ROI names tracks kept with another run."""
    rec = os.path.join(root, 'STAGE-13', 'rec-a-13')
    r = write_recording(os.path.join(rec, 'gene_expression'), SIGNALS, name=RESULTS_NAME)
    os.remove(r['tracklets'])
    shutil.rmtree(r['masks_dir'])
    with open(os.path.join(rec, 'roi_selection.json'), 'w') as f:
        json.dump({'t_first': 1, 't_last': 4,
                   'tracklets': os.path.join(tracks_rec, 'unified', 'tracklets_unified.json')}, f)
    return rec


def test_results_found_in_both_layouts():
    with tempfile.TemporaryDirectory() as root:
        a = pipeline_result(root)
        split_result(root, a)
        os.makedirs(os.path.join(root, '_cache'))          # never searched
        rows = {r['name']: r for r in discover([root])}
        assert set(rows) == {'rec-a', 'rec-a-13'}, rows
        for name, group in (('rec-a', 'STAGE-12'), ('rec-a-13', 'STAGE-13')):
            r = rows[name]
            assert r['problem'] is None, r
            assert r['group'] == group and (r['n_frames'], r['n_cells'], r['n_tracks']) == (6, 4, 4), r
            assert r['tracklets'] == os.path.join(a, 'unified', 'tracklets_unified.json')
            assert r['masks_dir'] == os.path.join(a, 'unified', 'masks_split')
            assert r['projection'] == os.path.join(a, 'preprocess', 'SUM_C2-rec-a.tif')
            assert (r['height'], r['width']) == (16, 16)
        assert rows['rec-a']['roi_window'] is None and rows['rec-a-13']['roi_window'] == [1, 4]
        assert find_matrices(os.path.join(root, 'STAGE-12')) == [
            os.path.join(a, 'gene_expression', RESULTS_NAME)]
    passed('results are found in the pipeline layout and with tracks kept elsewhere')


def test_all_cells_result_replaces_roi_ones():
    """A gene_expression_all/ result beside the tracks hides the results measured on the same
    tracks (here the pipeline's own and a split one); other tracks keep theirs."""
    with tempfile.TemporaryDirectory() as root:
        a = pipeline_result(root)
        split_result(root, a)
        pipeline_result(root, name='rec-b')
        write_recording(os.path.join(a, 'gene_expression_all'), SIGNALS, name=RESULTS_NAME)
        rows = [(r['name'], r['folder']) for r in discover([root])]
        assert sorted(rows) == [('rec-a', 'gene_expression_all'), ('rec-b', 'gene_expression')], rows
    passed('an all-cells result replaces the ROI-limited results on the same tracks')


def test_sample_cells_match_the_comparison():
    """Presence, volume CV, the mean position and the positions at the window's ends, for a
    window cutting into a cell's life."""
    with tempfile.TemporaryDirectory() as root:
        a = pipeline_result(root)
        rec = discover([root])[0]
        d = sample_data(rec, 1, 4, root)
        assert d['timepoints'] == [1, 2, 3, 4]
        cells = {c['cell']: c for c in d['cells']}
        assert set(cells) == {'cell_0', 'cell_1', 'cell_2', 'cell_3'}
        assert cells['cell_1']['presence'] == 0.75 and cells['cell_2']['presence'] == 0.75
        assert cells['cell_0']['vol_cv'] == 0                    # the fixture's tiles never change
        assert cells['cell_2']['at_start'] is None and cells['cell_2']['at_end'] is not None
        assert d['traces'][1] == [0.0, None, 0.0, 50.0]          # untracked frames are None
        # the mean position is the comparison's own
        with open(rec['tracklets']) as f:
            tracklets = json.load(f)
        props = cell_properties(rec['masks_dir'])
        names = ['cell_0', 'cell_1', 'cell_3']
        positions, _ = cell_positions(props, tracklets, names, np.arange(1, 5))
        for name, (y, x) in zip(names, positions):
            assert np.isclose(cells[name]['x'], x, atol=0.05) and np.isclose(cells[name]['y'], y, atol=0.05)
        assert os.path.exists(os.path.join(root, '_cache'))      # the masks are read once
        # the middle frame: halfway (halves up, as the page rounds) unless set, kept in the window
        from ms2.app.analysis import mid_frame
        from ms2.cell_activity_compare import cell_window
        assert d['t_mid'] == 3 and mid_frame(1, 4) == 3 and mid_frame(1, 5) == 3
        assert mid_frame(1, 4, 2) == 2 and mid_frame(1, 4, 9) == 4 and mid_frame(1, 4, 0) == 1
        d2 = sample_data(rec, 1, 4, root, t_mid=2)
        at2 = {c['cell']: c for c in d2['cells']}
        assert d2['t_mid'] == 2 and at2['cell_1']['at_mid'] is None and at2['cell_1']['label_mid'] is None
        w = cell_window(props, tracklets, names, np.arange(1, 5), t_mid=2)
        for name, (x, y) in zip(names, w['at_mid']):
            got = at2[name]['at_mid']
            assert (got is None) == np.isnan(x), name
            assert got is None or np.allclose(got, [x, y], atol=0.05), name
    passed('a window\'s cells: presence, volume CV, positions as the comparison has them, the middle frame too')


def test_ladder_cuts_reproduce_kmeans():
    """The rate cuts the page's sliders start from give exactly the levels KMeans fitted, and
    moving a cut moves only the cells it passes."""
    import pandas as pd
    from ms2.cell_activity import classify_cells, ladder_cuts, levels_from_cuts
    rng = np.random.default_rng(3)
    desc = pd.DataFrame({'rate': np.r_[np.zeros(30), rng.lognormal(0, 1.2, 300)],
                         'peak': np.r_[np.zeros(20), rng.uniform(1, 9, 10), rng.uniform(5, 200, 300)]})
    fitted = classify_cells(desc, score='rate')
    cuts = ladder_cuts(desc, fitted)
    assert len(cuts) == 2 and cuts[0] < cuts[1], cuts
    assert (levels_from_cuts(desc, cuts) == fitted).all()
    raised = levels_from_cuts(desc, [cuts[0], cuts[1] * 2])
    moved = raised != fitted
    assert moved.any() and (fitted[moved] == 4).all() and (raised[moved] == 3).all()
    passed('rate cuts reproduce the KMeans ladder, and a moved cut moves only the cells it passes')


def test_outline_files_and_label_image():
    """The ROI and segment files offered for a result, read back; and the label image the page
    outlines cells with decodes to the frame's labels projected over z."""
    import io
    from PIL import Image
    from ms2.app.analysis import label_png, outline_files, read_outline_file
    from ms2.viewer.loaders import MaskStore
    with tempfile.TemporaryDirectory() as root:
        a = pipeline_result(root)
        rec = discover([root])[0]
        tri = [[0, 0], [5, 0], [5, 5]]
        with open(os.path.join(a, 'roi_selection.json'), 'w') as f:
            json.dump({'t_first': 1, 't_last': 4, 'roi_first': tri, 'roi_last': None,
                       'tracklets': rec['tracklets']}, f)
        with open(os.path.join(a, 'segments.json'), 'w') as f:           # saved with the ROI, empty
            json.dump({'t': 1, 'segments': {}}, f)
        cmp_dir = os.path.join(root, 'cmp')
        os.makedirs(cmp_dir)
        with open(os.path.join(cmp_dir, 'recordings.json'), 'w') as f:
            json.dump({'A': {'csv': rec['csv']}, 'B': {'csv': '/elsewhere.csv'}}, f)
        with open(os.path.join(cmp_dir, 'segments.json'), 'w') as f:
            json.dump({'A': {'t': 2, 'orientation': {'flip': True, 'rotate': 90}, 'segments': {'3': tri}},
                       'B': {'t': 2, 'segments': {'2': tri}}}, f)
        files = outline_files(rec, [root])
        assert [r['path'] for r in files['rois']] == [os.path.join(a, 'roi_selection.json')], files
        assert files['segments'] == [{'path': os.path.join(cmp_dir, 'segments.json'), 'name': 'A'}], files
        roi = read_outline_file(files['rois'][0]['path'])
        assert roi == {'kind': 'roi', 'roi_first': tri, 'roi_last': None, 't_first': 1, 't_last': 4}
        seg = read_outline_file(files['segments'][0]['path'], 'A')
        assert seg['segments'] == {'3': tri} and seg['orientation'] == {'flip': True, 'rotate': 90}
        px = np.asarray(Image.open(io.BytesIO(label_png(rec['masks_dir'], 2)))).astype(int)
        assert (px[..., 0] * 256 + px[..., 1] == MaskStore(rec['masks_dir']).get(2).max(axis=0)).all()
    passed('ROI and segment files are offered per recording; the label image holds the labels')


def test_pipeline_measures_every_tracked_cell():
    """Without an ROI, every cell tracked for 10 frames is measured, also one born late or
    leaving early -- which the old rule (fewer than 20 missing frames) dropped."""
    from ms2.gene_expression.run import cells_to_measure
    tracklets = {'whole': [1] * 60,
                 'born_late': [-1] * 40 + [2] * 20,       # 40 frames before it appears
                 'leaves': [3] * 15 + [-2] * 45,          # leaves the field after 15
                 'blip': [4] * 9 + [-1] * 51}
    assert cells_to_measure(tracklets) == ['whole', 'born_late', 'leaves']
    passed('the pipeline measures every cell tracked 10 frames or more, born late or not')


def test_region_and_segments_saved_by_date():
    """A sample's ROI and segments go to <web>/region_and_segment/<today, DDMMYY>/<recording>_stage<N>.json
    with its start and end frames; a window outside the recording or with no frames is refused, and
    the files of every day are offered for loading back."""
    from ms2.app.analysis import is_region_file, outline_files, read_outline_file, resolve, save_region
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as web:
        day = os.path.join(root, '020626')
        pipeline_result(day)
        rec = resolve(discover([day])[0]['csv'])
        tri = [[0, 0], [8, 0], [0, 8]]
        sample = {'csv': rec['csv'], 'stage': '12', 't0': 1, 'tm': 3, 't1': 4, 'orientation': {'flip': True, 'rotate': 90},
                  'roi': {'first': tri, 'mid': tri, 'last': None}, 'segments': {'2': {'first': None, 'last': tri}}}
        got = save_region(rec, sample, web)
        import datetime
        path = os.path.join(web, 'region_and_segment', datetime.date.today().strftime('%d%m%y'), 'rec-a_stage12.json')
        assert got == {'path': path, 't_first': 1, 't_mid': 3, 't_last': 4}, got
        with open(path) as f:
            saved = json.load(f)
        assert (saved['t_first'], saved['t_mid'], saved['t_last'], saved['stage'], saved['csv']) == (1, 3, 4, '12', rec['csv'])
        assert saved['roi_first'] == tri and saved['roi_mid'] == tri and saved['roi_last'] is None
        assert saved['segments'] == {'2': {'first': None, 'mid': None, 'last': [[float(x), float(y)] for x, y in tri]}}
        # an ROI made from the segments: several outlines on a frame
        two = [tri, [[10, 10], [14, 10], [10, 14]]]
        save_region(rec, {**sample, 'roi': {'first': two, 'mid': None, 'last': None}}, web)
        with open(path) as f:
            assert json.load(f)['roi_first'] == two
        save_region(rec, sample, web)
        for bad in ({'t0': None}, {'t1': None}, {'tm': None}, {'tm': 0}, {'tm': 5}, {'t0': 5, 't1': 4},
                    {'t1': 10}, {'t0': -1}, {'roi': {'first': [tri, tri[:2]]}, 'segments': {}},
                    {'roi': {}, 'segments': {}}, {'stage': ''}):
            try:
                save_region(rec, {**sample, **bad}, web)
            except ValueError:
                continue
            raise AssertionError(f'{bad} was saved')
        # the same name kept for another result of the recording is left alone
        with open(path, 'w') as f:
            json.dump({**saved, 'csv': '/elsewhere/gene_expression_results.csv'}, f)
        try:
            save_region(rec, sample, web)
            raise AssertionError('another result\'s file was overwritten')
        except ValueError:
            pass
        with open(path, 'w') as f:
            json.dump(saved, f)
        assert is_region_file(path, web) and not is_region_file(rec['csv'], web)
        # a save from an earlier day is kept beside it, and offered too
        earlier = os.path.join(web, 'region_and_segment', '020626', 'rec-a_stage12.json')
        os.makedirs(os.path.dirname(earlier))
        shutil.copy(path, earlier)
        files = outline_files(rec, [day], web)
        for p in (path, earlier):
            assert {'path': p, 't_first': 1, 't_mid': 3, 't_last': 4, 'stage': '12'} in files['rois'], files
            assert {'path': p, 'name': None} in files['segments'], files
        assert saved['tracklets'] == rec['tracklets']
        # saved from another result on the same tracks (e.g. the ROI-limited one an all-cells
        # result replaced): offered too; on other tracks: not
        other_day = os.path.join(web, 'region_and_segment', '010126')
        os.makedirs(other_day)
        for name, tracks in (('same.json', rec['tracklets']), ('other.json', '/elsewhere/tracklets.json')):
            with open(os.path.join(other_day, name), 'w') as f:
                json.dump({**saved, 'csv': '/elsewhere/gene_expression_results.csv', 'tracklets': tracks}, f)
        offered = {r['path'] for r in outline_files(rec, [day], web)['rois']}
        assert os.path.join(other_day, 'same.json') in offered and os.path.join(other_day, 'other.json') not in offered
        back = read_outline_file(path)
        assert back['kind'] == 'region' and back['roi_first'] == tri and back['roi_mid'] == tri
        assert (back['t_first'], back['t_mid'], back['t_last']) == (1, 3, 4)
        assert back['segments'] == saved['segments']
    passed('ROI and segments are saved under <web>/region_and_segment/<today> with their frames, and offered back')


def test_only_results_measured_on_every_cell():
    """The list and the ms2_web browser take a result whose MS2 was measured on every cell: an
    all-cells result, or a pipeline run done without an ROI -- not an ROI-limited one.  The
    browser walks only the tree it is given."""
    from ms2.app.analysis import browse, measured_all_cells
    with tempfile.TemporaryDirectory() as ws:
        full, limited = (pipeline_result(os.path.join(ws, 'dana', '101126'), name=n) for n in ('full', 'limited'))
        for rec, roi in ((full, None), (limited, os.path.join(limited, 'roi_selection.json'))):
            with open(os.path.join(rec, 'run_manifest.json'), 'w') as f:
                json.dump({'steps_done': ['preprocess', 'segment', 'track', 'unify', 'gene_expression', 'activity'],
                           'params': {'roi': roi}}, f)
        csv = lambda rec: os.path.join(rec, 'gene_expression', RESULTS_NAME)
        assert measured_all_cells(csv(full)) and not measured_all_cells(csv(limited))
        assert [r['name'] for r in discover([ws], all_cells_only=True)] == ['full']
        assert {r['name'] for r in discover([ws])} == {'full', 'limited'}
        top = browse(ws, ws)
        assert top['parent'] is None and top['folders'] == ['dana'] and [r['name'] for r in top['results']] == ['full']
        assert top['results'][0]['group'] == os.path.join('dana', '101126', 'STAGE-12')   # the fixture's layout
        day = browse(os.path.join(ws, 'dana', '101126'), ws)
        assert day['parent'] == os.path.join(os.path.realpath(ws), 'dana') and day['folders'] == ['STAGE-12']
        for outside in (os.path.dirname(ws), os.path.join(ws, '..')):
            try:
                browse(outside, ws)
                raise AssertionError(f'{outside} browsed')
            except PermissionError:
                pass
    passed('only results measured on every cell are listed; the ms2_web browser stays inside it')


def test_roi_joins_the_segments():
    """The ROI is one outline per frame around the segments: two squares 6 px apart become one
    polygon holding both and the gap between them; a frame with one segment keeps its outline;
    a frame with none has no ROI."""
    from matplotlib.path import Path
    from ms2.app.analysis import joined_roi
    sq = lambda x0, y0: [[x0, y0], [x0 + 40, y0], [x0 + 40, y0 + 40], [x0, y0 + 40]]
    roi = joined_roi({'2': {'first': sq(10, 10), 'mid': sq(10, 10), 'last': None},
                      '3': {'first': sq(56, 10), 'mid': None, 'last': None}})
    assert roi['last'] is None and roi['mid'] == sq(10, 10), roi
    p = Path(roi['first'])
    assert all(isinstance(v, list) and len(v) == 2 for v in roi['first']) and len(roi['first']) >= 4
    assert p.contains_points([[15, 15], [45, 45], [53, 30], [90, 45]]).all()          # both, and the gap
    assert not p.contains_points([[5, 30], [100, 30], [53, 60], [53, 4]]).any()      # nothing much beyond
    passed('the ROI is the segments joined into one outline per frame')


def test_emitters_per_frame():
    """The page's MS2 emitters: each cell's accepted emitters (its _final table) by frame, at
    their image position, with the value the matrix holds for it."""
    from ms2.app.analysis import emitters
    with tempfile.TemporaryDirectory() as root:
        rec = pipeline_result(root)
        g = os.path.join(rec, 'gene_expression')
        with open(os.path.join(g, 'cell_3_data_global_peaks_final.csv'), 'w') as f:
            f.write('timepoint,x,y,intensity,ellipse_sum,noise\n1,10,4,30.0,31.04,1.5\n4,11,5,50.0,52.0,1.5\n')
        with open(os.path.join(g, 'cell_7_data_global_peaks_final.csv'), 'w') as f:
            f.write('timepoint,x,y,intensity,ellipse_sum,noise\n1,2,9,20.0,21.0,1.5\n')
        with open(os.path.join(g, 'cell_7_data_global_peaks.csv'), 'w') as f:       # before the filter: not read
            f.write('timepoint,x,y,intensity\n2,3,3,9.0\n')
        got = emitters(os.path.join(g, RESULTS_NAME))
        assert sorted(got) == [1, 4], got
        assert sorted(got[1]) == [['cell_3', 10.0, 4.0, 31.0], ['cell_7', 2.0, 9.0, 21.0]], got
        assert got[4] == [['cell_3', 11.0, 5.0, 52.0]], got
    passed("the accepted MS2 emitters are read per frame, the unfiltered peaks not")


def test_ms2_stack_found_in_both_layouts():
    """The background-subtracted MS2 stack shown beside the embryo: next to the cell-channel
    projection (_bg_removed, or the older hand-run _bg_removal), else in preprocess/."""
    from ms2.app.analysis import ms2_stack
    with tempfile.TemporaryDirectory() as root:
        data, rec_dir = os.path.join(root, 'data'), os.path.join(root, 'out', 'rec')
        os.makedirs(data); os.makedirs(os.path.join(rec_dir, 'preprocess'))
        proj = os.path.join(data, 'SUM_C2-rec.tif')
        rec = {'projection': proj, 'rec_dir': rec_dir}
        assert ms2_stack(rec) is None
        mine = os.path.join(rec_dir, 'preprocess', 'C1-rec_bg_removed.tif')
        open(mine, 'w').close()
        assert ms2_stack(rec) == mine
        old = os.path.join(data, 'C1-rec_bg_removal.tif')
        open(old, 'w').close()
        assert ms2_stack(rec) == old                 # the one beside the projection comes first
        assert ms2_stack({'projection': None, 'rec_dir': rec_dir}) == mine
    passed("the MS2 stack is found beside the embryo image or in preprocess/")


def test_save_load_and_export():
    """A single recording over two stages, saved, listed, loaded back and exported: the
    comparison's tables per stage and the segment maps, with the ladder set by hand."""
    import pandas as pd
    from ms2.app.analysis import (analysis_dir, export, list_analyses, load_analysis,
                                  save_analysis, sample_keys)
    rng = np.random.default_rng(0)
    signals = np.where(rng.random((12, 10)) < 0.4, 20 + rng.uniform(5, 80, (12, 10)), 0)
    with tempfile.TemporaryDirectory() as root:
        pipeline_result(root, signals, 'rec-b')
        rec = discover([root])[0]
        whole = [[0, 0], [16, 0], [16, 12], [0, 12]]           # the fixture's 4 x 3 tiles
        stage = lambda st, t0, t1, seg: {'csv': rec['csv'], 'name': 'rec-b', 'stage': st, 't0': t0, 't1': t1,
                                         'orientation': {'flip': False, 'rotate': 0}, 'segments': {'2': seg}}
        # stage 12 drawn on its start and end frames; stage 13 a single outline saved before that
        both = {'first': whole, 'mid': whole, 'last': whole}
        a = {'mode': 'single', 'stage': '12', 'samples': [stage('12', 0, 4, both), stage('13', 5, 9, whole)],
             'filters': {'presence': 0.9, 'vol_cv': None},
             'cluster': {'floor': 20, 'noise_peak': 10, 'cuts': [5.0, 20.0], 'auto': False},
             'box_from': 'cells', 'hist': {'mode': 'grid', 'size': 2}}
        for bad in ('', '../x', 'a/b'):
            try:
                save_analysis(root, bad, a)
            except ValueError:
                pass
            else:
                raise AssertionError(f'{bad!r} accepted as a name')
        # stage 12's ROI made from its segments: two outlines on the start frame
        a['samples'][0]['roi'] = {'first': [whole, [[0, 0], [1, 0], [0, 1]]], 'mid': None, 'last': None}
        save_analysis(root, 'two stages', a)
        assert [r['name'] for r in list_analyses(root)] == ['two stages']
        assert load_analysis(root, 'two stages')['samples'] == a['samples']
        assert sample_keys(a) == ['rec-b_stage12', 'rec-b_stage13']
        folder = analysis_dir(root, 'two stages')
        export(folder)
        cells = pd.read_csv(os.path.join(folder, 'cells.csv'))
        assert set(cells['recording']) == {'rec-b_stage12', 'rec-b_stage13'}
        for f in ('metrics.csv', 'heatmap_active_pooled.png', 'segments.json',
                  'segment_activity/segment_2_activity.png', 'segment_activity/segment_2_intensity.png',
                  'segment_activity/segment_2_clusters.png'):
            assert os.path.exists(os.path.join(folder, f)), f
        # the hand-set ladder: every active cell's level follows its rate and the two cuts
        active = cells[cells['peak'] >= 10]
        expect = 2 + (active['rate'] >= 5).astype(int) + (active['rate'] >= 20).astype(int)
        assert (active['level'] == expect).all()
        with open(os.path.join(folder, 'segment_activity', 'params.json')) as f:
            assert json.load(f)['box_from'] == 'cells'
        # the results zip: settings, outlines (drawn and as a file), figures and tables
        import io
        import zipfile
        from ms2.app.analysis import write_zip
        with open(os.path.join(folder, 'outlines.json')) as f:
            outlines = json.load(f)
        assert outlines['rec-b_stage12']['segments'] == {'2': both}
        # the old single outline is taken as drawn on the start frame, as the page takes it
        with open(os.path.join(folder, 'segments.json')) as f:
            segs = json.load(f)
        assert segs['rec-b_stage13']['segments'] == {'2': {'first': whole, 'last': None}}
        # a segment takes in the cells inside it on its frames: here every cell tracked on one
        members = pd.read_csv(os.path.join(folder, 'segment_activity', 'segment_cells.csv'))
        with open(os.path.join(folder, 'recordings.json')) as f:
            assert json.load(f)['rec-b_stage12']['t_mid'] == 2        # halfway through frames 0..4
        assert {'x_mid', 'y_mid'} <= set(cells.columns)
        # the ROI keeps the cells tracked on its frame, inside one of its outlines
        assert cells.loc[cells['recording'] == 'rec-b_stage12', 'x_start'].notna().all()
        for key, frames in (('rec-b_stage12', ('start', 'mid', 'end')), ('rec-b_stage13', ('start',))):
            c = cells[cells['recording'] == key]
            seen = np.zeros(len(c), dtype=bool)
            for w in frames:
                seen |= c[f'x_{w}'].notna().to_numpy()
            assert set(members.loc[members['recording'] == key, 'cell']) == set(c['cell'][seen]), key
        buf = io.BytesIO()
        write_zip(folder, buf)
        names = zipfile.ZipFile(buf).namelist()
        # the main results at the top; settings, outlines, GIFs and the rest in work_product/
        top = {n for n in names if not n.startswith('two stages/work_product/')}
        assert top == {'two stages/heatmap_active_pooled.png', 'two stages/level_composition.png',
                       'two stages/spatial_maps_aligned.png'} | {
            n for n in names if n.startswith('two stages/segment_activity/')}, sorted(top)
        assert not any(n.endswith('.gif') for n in top)
        for want in ('analysis.json', 'outlines.json', 'outlines_rec-b_stage12.png', 'cells.csv',
                     'heatmap_active_rec-b_stage12.png', 'segment_activity/segment_2_activity.gif'):
            assert 'two stages/work_product/' + want in names, want
        assert not any(n.endswith('.eps') or '/cache/' in n or '/tracklets/' in n for n in names), names
    passed('an analysis saves, lists, loads back and exports its tables and maps')


def main():
    for name, test in sorted(globals().items()):
        if name.startswith('test_'):
            test()
    print('analysis: all tests passed')


if __name__ == '__main__':
    main()
