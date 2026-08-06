"""Streaming cell tracker.

Performs the same IoU/Hungarian pairwise matching + tracklet construction as the
batch routine in ``cell_tracking.cell_tracking``, but consumes segmentation masks
one frame at a time so it can run *concurrently* with segmentation: as soon as two
consecutive frames are available the pair is matched, and tracklets are assembled
once the final frame arrives.

This module is intentionally light (no torch/cellpose) so it can be imported in the
pipeline orchestrator and in unit tests without loading the segmentation stack.
"""

import os
import json

import numpy as np

from src.track import (
    match_cells_by_iou_hungarian_local_optimized,
    create_tracklets,
)


class StreamingTracker:
    """Incrementally match cells between consecutive frames, then build tracklets.

    Usage::

        tracker = StreamingTracker()
        for ti, mask_path in frames_in_order:      # ascending, consecutive ti
            tracker.add_frame(ti, mask_path)
        tracklets = tracker.finalize(save_path=...)

    The produced ``tracklets`` are identical to running the batch pipeline
    (``match_over_time_cell_iou`` followed by ``create_tracklets``) over the same
    masks, because the per-pair matches are accumulated in frame order.
    """

    def __init__(self, match_fn=None):
        # Defaults to the same matcher used by cell_tracking.match_over_time_cell_iou.
        self._match_fn = match_fn or match_cells_by_iou_hungarian_local_optimized
        self._prev_ti = None
        self._prev_mask = None
        self.matched_points = []    # ordered: one dict per consecutive frame pair
        self.seen_timepoints = []   # ordered list of timepoints added

    @staticmethod
    def _load_mask(mask_path):
        with np.load(mask_path, allow_pickle=True) as data:
            return data['masks']

    def add_frame(self, ti, mask_path):
        """Register a newly-available frame.

        Frames must be supplied in ascending order. When a frame is exactly one
        step after the previous one, its match against the previous frame is
        computed immediately and appended to ``matched_points``.

        Args:
            ti (int): Timepoint index of this frame.
            mask_path (str): Path to the ``.npz`` holding this frame's ``masks``.

        Returns:
            bool: True if a new consecutive-pair match was produced.
        """
        ti = int(ti)
        mask = self._load_mask(mask_path)
        produced = False

        if self._prev_mask is not None:
            if ti == self._prev_ti + 1:
                matches = self._match_fn(self._prev_mask, mask)
                self.matched_points.append(matches)
                produced = True
            elif ti <= self._prev_ti:
                # Duplicate or out-of-order frame: ignore it, keep current state.
                return False
            else:
                # A gap means an intermediate mask is missing. create_tracklets
                # assumes a contiguous chain of pairwise matches, so refuse to
                # silently corrupt the tracks.
                raise ValueError(
                    f"Non-consecutive frame {ti} after {self._prev_ti}; "
                    "an intermediate mask is missing."
                )

        self._prev_ti = ti
        self._prev_mask = mask
        self.seen_timepoints.append(ti)
        return produced

    def finalize(self, save_path=None):
        """Build tracklets from all accumulated pairwise matches.

        Args:
            save_path (str | None): If given, write the tracklets as JSON here
                (same format/location convention as ``cell_tracking.cell_tracking``).

        Returns:
            dict: Tracklets mapping tracklet id -> list of per-timepoint labels.
        """
        tracklets = create_tracklets(self.matched_points)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(tracklets, f, indent=4)
            print(f"Tracklets saved to {save_path}")
        return tracklets
