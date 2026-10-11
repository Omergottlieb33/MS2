Tracking fixes, 2026-09-24
=========================

Interior disappearances are treated as unresolved missing observations, not biological deaths. No synthetic cell masks are invented to fill gaps. The legacy `death_frame` field in diagnostics denotes the last observed frame only.

Implemented changes:

- Exact full-volume IoU replaces truncated centroid-centered intersections.
- One continuous overlap/distance cost; scaled 3D distance is the default.
- Optional Hungarian assignment with explicit unmatched costs; reject invalid candidates before assignment.
- Initialize tracks from every detection, including t0 and single-frame detections.
- Recompute one-frame gap recovery against missing tracks and free targets. Use a larger, gap-dependent distance gate.
- Annotate possible XY exits after recovery, using mask contact. Z-only disappearance remains unresolved in this shallow stack.
- Stitch selected chains in reverse time so A→B→C survives donor deletion. Add gap and volume penalties, optional assignment, and scaled 3D gating.
- Respect frame limits, handle t=None, and reject missing/unreadable frames rather than compressing time.
- Load CZI dependencies only when opening a CZI image, allowing mask-only processing in the available environment.

Validation commands:

```bash
PYTHONPATH=. /home/dafei/miniconda3/envs/ultrack/bin/python tests/test_tracking.py
```

Ten regression tests cover the fixes. The full 72-frame New-03 validation output is in:

`/zjbd/zd1/shechtmanlab/omer/MS2/outputs/020626/STAGE-12/New-03-V/New-03-V/tracking_fixed_20260924/`

Use `unified/tracklets_unified.json` with `unified/masks_split/` for the new unified result. Use `tracklets_default.json` with the original `masks/` for the corrected default result. Track IDs are newly generated and should not be compared numerically with old IDs.

`comparison.md` and `comparison.json` compare the previous unified output with both new outputs. `run_manifest.json` and `source_snapshot/` record parameters, versions, input tracklet hashes, and the implementation used. `ms2/compare_tracking.py` performs the reproducible comparison and checks mask/track consistency.

Preserving single-frame detections increases track counts. Compare larger-cell continuity, coverage, and candidate identity errors as well as track duration. Visual validation is still needed before judging biological identity accuracy.

Completed comparison
--------------------

See [TRACKING_COMPARISON.md](TRACKING_COMPARISON.md) for the measured old/new comparison. Zero-overlap consecutive links fell from 179 to 7 and heuristic jump flags from 787 to 525. Larger-cell mean active duration decreased from 58.30 to 57.69 frames; tracks present in every frame decreased from 299 to 236. These are mixed diagnostics, not a demonstrated identity-accuracy improvement. The new unified output retains 526 single-frame tracks, explaining much of the increase in total track count. All outputs passed mask-label existence and unique-ownership checks.
