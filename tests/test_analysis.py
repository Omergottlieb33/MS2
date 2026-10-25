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
    passed('a window\'s cells: presence, volume CV, positions as the comparison has them')


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
        stage = lambda st, t0, t1: {'csv': rec['csv'], 'name': 'rec-b', 'stage': st, 't0': t0, 't1': t1,
                                    'orientation': {'flip': False, 'rotate': 0}, 'segments': {'2': whole}}
        a = {'mode': 'single', 'stage': '12', 'samples': [stage('12', 0, 4), stage('13', 5, 9)],
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
        assert outlines['rec-b_stage12']['segments'] == {'2': whole}
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
