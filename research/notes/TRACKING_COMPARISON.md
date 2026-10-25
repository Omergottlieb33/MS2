Tracking validation: New-03, 72 frames

| Metric | Previous unified | Corrected default | Corrected unified |
|---|---:|---:|---:|
| tracks | 1230 | 2056 | 1815 |
| full size tracks | 689 | 737 | 693 |
| full size mean active | 58.30 | 53.70 | 57.69 |
| mean active frames | 38.74 | 23.59 | 26.64 |
| median active frames | 39.00 | 5.00 | 8.00 |
| full movie tracks | 299 | 245 | 236 |
| singleton tracks | 0 | 661 | 526 |
| active detections | 47647 | 48506 | 48346 |
| unowned detections | 1203 | 0 | 673 |
| unresolved interior endings | 759 | 1570 | 1331 |
| internal gap frames | 378 | 551 | 553 |
| zero overlap links | 179 | 6 | 7 |
| heuristic jump flags | 787 | 521 | 525 |
| duplicate claims | 0 | 0 | 0 |
| absent labels | 0 | 0 | 0 |

No biological deaths are assumed. Interior endings are unresolved identity/detection losses.
Possible exits use the same XY mask-contact rule for every output; Z exits remain unresolved.
Tracks include fragments and single-frame detections. Counts are not identity accuracy.
Full-size subgroup uses mean segmented volume >= 600 voxels for all outputs; this is a heuristic, not ground truth.
Unified masks differ. More unowned objects can result from deliberate fragment retirement.
Zero overlap and jump flags are inspection candidates, not confirmed identity errors.

Visualization inputs:

previous_unified
- Tracklets: /zjbd/zd1/shechtmanlab/omer/MS2/outputs/020626/STAGE-12/New-03-V/New-03-V/unify/tracklets_unified.json
- Masks: /zjbd/zd1/shechtmanlab/omer/MS2/outputs/020626/STAGE-12/New-03-V/New-03-V/unify/masks_split

corrected_default
- Tracklets: /zjbd/zd1/shechtmanlab/omer/MS2/outputs/020626/STAGE-12/New-03-V/New-03-V/tracking_fixed_20260924/tracklets_default.json
- Masks: /zjbd/zd1/shechtmanlab/omer/MS2/outputs/020626/STAGE-12/New-03-V/New-03-V/masks

corrected_unified
- Tracklets: /zjbd/zd1/shechtmanlab/omer/MS2/outputs/020626/STAGE-12/New-03-V/New-03-V/tracking_fixed_20260924/unified/tracklets_unified.json
- Masks: /zjbd/zd1/shechtmanlab/omer/MS2/outputs/020626/STAGE-12/New-03-V/New-03-V/tracking_fixed_20260924/unified/masks_split
