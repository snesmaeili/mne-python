"""Record of the temporal alignment applied to ragged epochs."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AlignmentRecord:
    """Description of how epochs were mapped onto a common time axis.

    Attached to the result of an alignment so that the transformation remains
    inspectable. In particular ``original_duration`` keeps the experimental
    trial durations available after a warp, and ``warps_spectral_content``
    reports whether the frequency axis of a subsequent time-frequency analysis
    can be interpreted at face value.

    Parameters
    ----------
    method : str
        One of ``'none'``, ``'common-crop'``, ``'pad'``,
        ``'duration-normalize'`` or ``'piecewise-linear'``.
    domain : str
        ``'signal'`` if the alignment was applied to the time-domain data,
        ``'tfr'`` if it was applied to a time-frequency representation.
    target_coord : str
        ``'seconds'`` or ``'phase'``, the units of the resulting axis.
    interpolation : str
        The interpolation used, currently always ``'linear'``.
    original_duration : array | None
        Duration of each epoch in seconds before alignment.
    original_start : array | None
        Start time of each epoch in seconds before alignment.
    original_end : array | None
        End time of each epoch in seconds before alignment.
    original_landmarks : list of array | None
        Landmark latencies of each epoch in seconds relative to its start.
    target_landmarks : array | None
        The common landmark latencies that every epoch was mapped onto.
    target_rule : str | None
        How ``target_landmarks`` was chosen, for example ``'median'``.
    landmark_names : tuple of str | None
        Names of the landmarks, in order.

    Notes
    -----
    .. versionadded:: 1.12
    """

    method: str
    domain: str
    target_coord: str
    interpolation: str = "linear"
    original_duration: np.ndarray | None = None
    original_start: np.ndarray | None = None
    original_end: np.ndarray | None = None
    original_landmarks: list | None = field(default=None)
    target_landmarks: np.ndarray | None = None
    target_rule: str | None = None
    landmark_names: tuple | None = None

    @property
    def warps_spectral_content(self):
        """Whether this alignment shifts apparent frequency.

        Rescaling the time axis of a signal before a time-frequency transform
        rescales the frequencies of the oscillations it carries: a 10 Hz
        oscillation stretched by a factor of two is reported near 5 Hz.
        Applying the same warp to a time-frequency representation instead moves
        energy along the time axis and leaves the frequency axis alone.
        """
        return self.domain == "signal" and self.method in (
            "duration-normalize",
            "piecewise-linear",
        )

    def summary(self):
        """Summarize the alignment in one line.

        Returns
        -------
        summary : str
            A human-readable description.
        """
        parts = [f"{self.method} in {self.domain} domain -> {self.target_coord}"]
        if self.target_rule:
            parts.append(f"target={self.target_rule}")
        if self.landmark_names:
            parts.append(f"landmarks={'/'.join(self.landmark_names)}")
        if self.warps_spectral_content:
            parts.append("apparent frequency is rescaled")
        return "; ".join(parts)

    def __repr__(self):
        """Build a string representation of the instance.

        Returns
        -------
        repr : str
            The representation.
        """
        return f"<AlignmentRecord | {self.summary()}>"
