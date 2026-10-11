Tracking review — 2026-09-24
============================

The default pipeline has fixable matching and track-construction problems. Ultrack's current integration does not directly address two central failure modes: cells absent from the foreground for a frame, and fused labels lacking an internal boundary. Its division model also changes what a tracklet count means. Neither fewer tracklets nor longer tracks establishes correct identities.

Scope and evidence
------------------

Reviewed `cell_tracking.py`, `src/track.py`, `src/track_unify.py`, `src/track_diagnostics.py`, `src/track_ultrack.py`, and installed Ultrack 0.7.2 source. Read all 72 original masks for New-03-V (shape 10×1024×1024), checked cached label identities and volumes against those masks, and audited saved tracks. Cache label/volume checks all passed. Unified statistics use the existing corrected-mask property cache. Examined the saved New-02 Ultrack database read-only. No tracking implementation or input data was changed; no full tracker rerun or ground-truth accuracy benchmark was performed.

New-03 root: `/zjbd/zd1/shechtmanlab/omer/MS2/outputs/020626/STAGE-12/New-03-V/New-03-V/`.

| Saved New-03 output | Default (`masks/tracklets_v2.json`) | Unified (`unify/tracklets_unified.json`) |
|---|---:|---:|
| Tracklets | 1,347 | 1,230 |
| Mean active frames | 35.57 | 38.74 |
| Median active frames | 28 | 39 |
| Tracks present at all 72 frames | 303 | 299 |
| Non-border track endings before final frame | 878 | 762 |
| Detections claimed by tracks | 47,908 | 47,647 |
| Mask detections not claimed by tracks | 598 | 1,203 |
| Duplicate ownership / references to absent labels | 0 / 0 | 0 / 0 |

These are descriptive, not accuracy scores. Unified masks have additional split objects; deliberate fragment retirement also increases unclaimed detections. The log contains 344 frame-level clump splits, 60 fragment retirements, 43 clump-resolution splices, 14 gap stitches, and 787 possible-swap flags. Those flags are heuristics, not confirmed swaps.

The 48,506 original detections have median volume 1,190 voxels; 8,119 (16.7%) are below 400 and 36 exceed 4,000. Across consecutive masks, the existing 25%-overlap criterion identifies 274 split-like and 312 merge-like events. These can reflect segmentation changes, motion, or biology; they are not annotated biological divisions/mergers. Of 46,195 saved consecutive default links, 192 have exactly zero full-volume overlap.

Default tracker findings, in priority order
------------------------------------------

1. **The cost function rewards losing the last bit of overlap.** `src/track.py:552–566` uses `0.7*(1-IoU)+0.3*d` for overlapping candidates and `0.5+0.5*d` for non-overlapping ones, where `d` is normalized distance. At equal distance zero, IoU 0.1 costs 0.63 while IoU zero costs only 0.5. This is a discontinuity in the wrong direction. A synthetic 3D example reproduced a zero-overlap match to another Z plane instead of the available overlapping candidate. Use one continuous cost formula; allow zero-overlap recovery only with strong motion/appearance evidence.

2. **Hungarian assignment has no explicit option to leave a plausible-but-poor pair unmatched.** `src/track.py:528–578` completes a rectangular assignment, then rejects invalid pairs. A finite forbidden cost of 2 does not solve the modeling problem; bad candidates can influence which good candidates get assigned. `min_iou` is also checked after assignment. Add dummy birth/death assignments with calibrated costs, and exclude forbidden pairs before solving. Raising `min_iou` alone leaves both issues in place.

3. **Track births are constructed from matches instead of detections.** `src/track.py:210–215,259–264` initializes only matched pairs and creates later tracks only from match dictionary entries. Singleton detections disappear from the output; a cell visible at t=0, absent at t=1, and back at t=2 cannot use the initial skip match because it never acquired a track. Initialize from every detection and explicitly manage active, temporarily missing, and terminated tracks. The 598 unclaimed New-03 detections demonstrate a coverage deficit, though they are not necessarily all real cells.

4. **XY-only gates permit cross-Z identity swaps.** `src/track.py:516–520` ignores Z by default; the alternative uses unscaled voxel coordinates. Use physical spacing `(0.5, 0.198, 0.198)` µm, or `(2.52,1,1)` in XY-pixel units, with a separately calibrated Z uncertainty if axial segmentation is unstable. Ignoring Z is not an anisotropy correction.

5. **Skip matching is too limited and competes in the wrong order.** `cell_tracking.py:79` and `src/track.py:242` allow exactly one missed frame, use the same distance gate for one- and two-frame intervals, and give immediate links priority over recovering missing tracks. Skip assignments are computed over all detections before knowing which tracks need recovery. Use joint candidate assignment over active and recently missing tracks, a motion prediction, and a gap-dependent uncertainty/cost. Avoid simply widening a nearest-neighbor radius in crowded tissue.

6. **Gap stitching loses selected chains.** `src/track_unify.py:467–476` deletes a donor while iterating assignments computed beforehand. A verified synthetic chain with detections at frames 0, 2, and 4 at identical positions yielded A+B and separate C: after A absorbs B, the chosen B→C edge is skipped. Assemble whole directed chains first, then materialize merged tracks, or maintain a representative map. Its `dxy/dt` score also favors longer gaps at equal displacement, with no penalty for missing frames; the gate reaches 35 XY pixels at dt=5. Add gap and uncertainty penalties and appearance/volume consistency.

7. **The unifier's segmentation assumptions need confidence checks.** Fixed median-volume thresholds cannot reliably distinguish a real partial nucleus, a large nucleus, a fragment, and a clump. Prior-frame watershed seeds can drift or propagate a wrong identity through multiple frames. Erode and motion-warp markers where appropriate, use raw nuclear intensity/boundary evidence, require plausible individual piece volumes, and check identities both before and after the clump. The documented synthetic split-recovery percentage does not validate moving, noisy real clumps.

8. **Border and evaluation rules hide failure modes.** `get_border_labels` considers only XY centroid proximity and treats a missed edge cell as permanently exited; Z departures are not classified. Use mask contact, trajectory, and an explicit uncertain-exit state, including Z when appropriate to the acquisition. `evaluate_tracklets` measures internal gaps only: one true trajectory split into many gap-free tracklets can have a perfect zero fragmentation index. Report coverage, interior starts/ends, manually checked ID switches, and lineage-aware metrics.

Additional implementation issues: `cell_tracking(masks_dir,t)` ignores `t` when loading/matching and raises for documented `t=None`; failed mask loads silently compress the time axis. Local intersection crops can omit overlap for large/elongated cells while retaining full object volumes in the union. Use the exact joint-label histogram already demonstrated by `split_merge_census`, plus a spatial index for zero-overlap candidates. These changes improve both correctness and speed.

Why Ultrack did not necessarily improve tracking
-----------------------------------------------

**The label input does not restore lost segmentation evidence.** The wrapper sends a single hard label stack (`src/track_ultrack.py:286`). Ultrack converts it into `labels > 0` foreground and label boundaries. An undetected cell has no foreground support; a fused label has no measured internal boundary. A hierarchy can still generate alternatives, but the correct biological split is not supplied by this conversion. The wrapper's opening claim that Ultrack can simply decline clumps is too strong. Supply complementary segmentations and/or image-derived boundary evidence, and inspect whether correct alternatives actually exist before optimizing tracking. See [Ultrack label conversion](https://github.com/royerlab/ultrack/blob/main/ultrack/utils/edge.py).

**There is no gap recovery in this path.** The installed linking implementation connects t only to t+1, confirmed directly in the saved database: all 164,560 candidate edges have time difference exactly one. Global optimization cannot cross a missing detection without an appropriate candidate path. The wrapper adds no post-solve stitching. Your baseline therefore has a recovery capability absent from this integration. See [Ultrack linking implementation](https://github.com/royerlab/ultrack/blob/main/ultrack/core/linking/processing.py).

**Divisions are inexpensive and tracklet counts are not directly comparable.** The current wrapper leaves Ultrack 0.7.2 defaults unchanged: appearance, disappearance, and division weights each -0.001; link transform `IoU**4`; distance weight 0; up to five selected candidate neighbors per target. A segmentation split can be explained as a biological division. The saved New-02 lineage has 725 parents and 1,450 daughter tracks. These are inferred, not validated, divisions. If division is biologically absent in the recording interval, prohibit it where supported or strongly penalize it and validate the result. If it occurs, score correct lineage events rather than treating all new daughter IDs as failures. See [tracking defaults](https://github.com/royerlab/ultrack/blob/main/ultrack/config/trackingconfig.py).

**The gate, size priors, and temporal horizon require validation.** A radius of 12 after applying `(2.52,1,1)` is a scaled 3D radius, whereas the cited motion percentile was measured in XY. Existing-track motion estimates are also censored by the old gate and possible mistaken links; they are not unbiased motion ground truth. On New-03, 269 saved consecutive default links exceed this scaled radius, though some may be incorrect. The area settings shape hypothesis eligibility; below-minimum isolated components and oversegmentation behavior mean they should not be interpreted as simple rejection counts. The default solve is in 25-frame windows with one-frame overlap, not one whole-movie optimization. Neither the number 12 nor the 400–4,000 voxel range should be transferred without checking candidate recall.

**The current wrapper does not use the intensity/motion support available to Ultrack.** It passes neither `images` nor a vector field. The saved database has zero nonzero motion shifts. The contours mode uses raw-image foreground/inversion, but quality depends on the nuclear signal and preprocessing; it is not automatically superior to Cellpose. `detect_contours` also uses the module constant for anisotropy rather than the CLI override (`src/track_ultrack.py:145`).

Saved New-02 evidence and limits
-------------------------------

Root: `/zjbd/zd1/shechtmanlab/omer/MS2/outputs/020626/STAGE-11/New-02-v3-ST11-12/`.

| Saved 109-frame output | Tracklets | Mean / median active frames | Present all frames | Active detections |
|---|---:|---:|---:|---:|
| `New-02-v3-ST11-12/masks/tracklets_bug_fix2.json` | 2,464 | 29.68 / 18 | 89 | 73,130 |
| `New-02-v3-ST11-12/unified/tracklets_unified.json` | 2,187 | 33.62 / 23 | 88 | 73,525 |
| `ultrack/tracklets_ultrack.json` | 3,341 | 19.32 / 9 | 37 | 64,554 |

The baseline file is an existing named variant, not a verified run of today's code. These outputs use different segmentations and division semantics. There is no full saved configuration/command manifest, so this is not a controlled benchmark. The Ultrack database selects 4,631 nodes smaller than 400 voxels and 680 larger than 4,000 (selected areas range 101–10,527); do not assume the saved run used today's wrapper bounds. No Ultrack output was found for the supplied New-03 dataset.

The Ultrack environment also lacks `czifile`, which prevents importing the current wrapper through its transitive diagnostic/viewer imports. This is a present reproducibility issue, not an explanation for the quality of an already completed run. Avoid requiring image-viewer/CZI dependencies just to track existing masks. `environment.yml` does not declare Ultrack either; save a dedicated, pinned environment and complete run configuration.

Recommended next experiment
---------------------------

1. Fix the cost discontinuity, unmatched assignments, detection-based track initialization, and stitching-chain bug first. Add targeted regression cases for these failures, not just tests of track length/schema.
2. Annotate a small set of trajectories and difficult events in raw 3D imagery: temporary loss, fused cells, oversegmentation, close neighbors in Z, real exits, and any real divisions. Use separate tuning and evaluation examples.
3. On identical frames/ROIs, compare corrected default tracking; default plus conservative segmentation repair; current label-only Ultrack; Ultrack with complementary segmentation/boundary evidence; and the latter with validated gap stitching. Save input hashes, spacing, all configuration values, versions, solver status/objective gap, and lineage.
4. Prioritize identity correctness and coverage. Report switches, missed detections, false joins, erroneous divisions, non-border breaks, and lineage-aware accuracy where annotations permit. Use track counts and duration as secondary diagnostics.

The strongest near-term opportunity is correcting the existing assignment/track lifecycle and supplying better evidence at segmentation failures. A more global solver helps only when its candidate graph contains the right cells and links.
