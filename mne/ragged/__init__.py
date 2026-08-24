"""Epochs of unequal duration.

Trials of different length arise whenever the process under study lasts as long
as it lasts: a gait cycle, a self-paced decision, a spoken sentence. This module
stores such trials at their true durations, supports the operations that are
mathematically per-trial, and requires an explicit choice before any analysis
that needs a common time axis.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from ._align import (
    align_time,
    common_crop,
    duration_normalize,
    landmark_warp,
    pad,
    piecewise_linear_warp,
    resolve_target_landmarks,
)
from ._container import RaggedEpochs, RaggedTimesError
from ._ops import (
    apply_baseline,
    compute_covariance,
    concatenate_for_decomposition,
    filter_epochs,
    map_epochs,
    set_eeg_reference,
)
from ._provenance import AlignmentRecord
from ._tfr import RaggedEpochsTFR, compute_tfr, warp_tfr

__all__ = [
    "AlignmentRecord",
    "RaggedEpochs",
    "RaggedEpochsTFR",
    "RaggedTimesError",
    "align_time",
    "apply_baseline",
    "common_crop",
    "compute_covariance",
    "compute_tfr",
    "concatenate_for_decomposition",
    "duration_normalize",
    "filter_epochs",
    "landmark_warp",
    "map_epochs",
    "pad",
    "piecewise_linear_warp",
    "resolve_target_landmarks",
    "set_eeg_reference",
    "warp_tfr",
]
