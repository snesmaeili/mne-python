"""Time-frequency representation of ragged epochs."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ..time_frequency import EpochsTFRArray, tfr_array_morlet
from ..utils import _check_option, logger, verbose
from ._align import _warp_complex, piecewise_linear_warp, resolve_target_landmarks
from ._container import RaggedTimesError
from ._provenance import AlignmentRecord


def _wavelet_length(freq, n_cycles, sfreq):
    """Return the length of a Morlet wavelet in samples.

    Parameters
    ----------
    freq : float
        Frequency in Hz.
    n_cycles : float
        Number of cycles.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    n_samples : int
        Length of the wavelet in samples.
    """
    sigma_t = n_cycles / (2.0 * np.pi * freq)
    return 2 * int(sigma_t * 5.0 * sfreq) + 1


def _check_wavelets_fit(freqs, n_cycles, sfreq, n_min):
    """Check that every wavelet fits inside the shortest epoch.

    Parameters
    ----------
    freqs : array
        Frequencies of interest in Hz.
    n_cycles : float | array
        Number of cycles per frequency.
    sfreq : float
        Sampling frequency in Hz.
    n_min : int
        Number of samples in the shortest epoch.
    """
    n_cycles = np.broadcast_to(np.asarray(n_cycles, dtype=float), freqs.shape)
    lengths = np.array([_wavelet_length(f, c, sfreq) for f, c in zip(freqs, n_cycles)])
    bad = lengths > n_min
    if not bad.any():
        return
    ok = freqs[~bad]
    if len(np.unique(lengths)) == 1:
        remedy = (
            "n_cycles is proportional to frequency here, so all wavelets have "
            "the same length and raising fmin will not help. Lower n_cycles, "
            "add context, or drop the shortest epochs."
        )
    elif ok.size:
        remedy = (
            f"Raise fmin to at least {ok.min():g} Hz, lower n_cycles, add "
            "context, or drop the shortest epochs."
        )
    else:
        remedy = (
            "No requested frequency fits. Lower n_cycles, add context, or drop "
            "the shortest epochs."
        )
    raise ValueError(
        f"{bad.sum()} of {len(freqs)} requested frequencies need a wavelet "
        f"longer than the shortest epoch ({lengths.max()} > {n_min} samples). "
        "The shortest epoch bounds the whole set, because every epoch must "
        f"yield the same frequency axis. {remedy}"
    )


class RaggedEpochsTFR:
    """Time-frequency representation of epochs with unequal duration.

    The frequency axis is common to all epochs; only the time axis is ragged.
    Averaging across epochs is undefined in this state, because a given time
    index does not correspond to the same point of the trial in every epoch,
    and raises :class:`~mne.ragged.RaggedTimesError`.

    Parameters
    ----------
    data : list of array
        Per-epoch data of shape ``(n_channels, n_freqs, n_times_i)``.
    info : instance of Info
        The measurement info.
    freqs : array
        The frequencies in Hz.
    tmin : float | array of float
        Start time of each epoch relative to its event, in seconds.
    output : str
        ``'power'``, ``'phase'`` or ``'complex'``.
    alignment : instance of AlignmentRecord | None
        Record of the alignment applied, if any.
    events : array of int, shape (n_epochs, 3) | None
        The events.
    metadata : instance of pandas.DataFrame | None
        Per-epoch metadata.
    sfreq : float | None
        Sampling frequency along the time axis, in Hz. Defaults to
        ``info["sfreq"]``.

    Attributes
    ----------
    info : instance of Info
        The measurement info.
    freqs : array
        The frequencies in Hz.
    output : str
        The kind of data held.
    alignment : instance of AlignmentRecord | None
        Record of the alignment applied, if any.
    events : array of int, shape (n_epochs, 3) | None
        The events.
    metadata : instance of pandas.DataFrame | None
        Per-epoch metadata.

    Notes
    -----
    .. versionadded:: 1.12
    """

    def __init__(
        self,
        data,
        info,
        freqs,
        tmin,
        *,
        output="power",
        alignment=None,
        events=None,
        metadata=None,
        sfreq=None,
    ):
        self._data = list(data)
        self.info = info
        self.freqs = np.asarray(freqs, dtype=float)
        self._tmin = np.broadcast_to(
            np.asarray(tmin, dtype=float), (len(self._data),)
        ).copy()
        self.output = output
        self.alignment = alignment
        self.events = events
        self.metadata = metadata
        self._sfreq = float(sfreq if sfreq is not None else info["sfreq"])

    def __len__(self):
        """Return the number of epochs.

        Returns
        -------
        n_epochs : int
            The number of epochs.
        """
        return len(self._data)

    @property
    def lengths(self):
        """Number of time samples in each epoch."""
        return np.array([d.shape[-1] for d in self._data], dtype=np.int64)

    @property
    def sfreq(self):
        """The sampling frequency along the time axis, in Hz."""
        return self._sfreq

    @property
    def is_uniform(self):
        """Whether all epochs share one time axis."""
        return bool(
            len(np.unique(self.lengths)) == 1 and np.allclose(self._tmin, self._tmin[0])
        )

    @property
    def times(self):
        """The time axis shared by all epochs, in seconds.

        Raises
        ------
        RaggedTimesError
            If the epochs have different durations or time origins.
        """
        if self.is_uniform:
            return self._tmin[0] + np.arange(self.lengths[0]) / self._sfreq
        raise RaggedTimesError(
            f"These {len(self)} time-frequency representations do not share a "
            "time axis. Use .get_times(i), or mne.ragged.warp_tfr(...) to put "
            "them on a common landmark-referenced axis."
        )

    def get_times(self, epoch=None):
        """Return the time vector of one or all epochs.

        Parameters
        ----------
        epoch : int | None
            Index of the epoch. If ``None``, return one vector per epoch.

        Returns
        -------
        times : array | list of array
            Time vector(s) in seconds.
        """
        if epoch is None:
            return [self.get_times(ii) for ii in range(len(self))]
        return self._tmin[epoch] + np.arange(self.lengths[epoch]) / self._sfreq

    def get_data(self, epoch=None):
        """Return the data.

        Parameters
        ----------
        epoch : int | None
            Index of a single epoch. If ``None``, return a list of arrays.

        Returns
        -------
        data : array | list of array
            The data, of shape ``(n_channels, n_freqs, n_times_i)`` per epoch.
        """
        return self._data[epoch] if epoch is not None else list(self._data)

    def apply_baseline(self, mode="logratio", baseline=None):
        """Apply a per-epoch baseline correction.

        Parameters
        ----------
        mode : str
            One of ``'logratio'``, ``'ratio'`` or ``'mean'``.
        baseline : tuple of float | None
            The baseline window in seconds. If ``None``, the whole epoch is
            used, which is the single-trial normalization commonly applied to
            cyclic data where no pre-trial period exists.

        Returns
        -------
        tfr : instance of RaggedEpochsTFR
            The baseline-corrected data.
        """
        _check_option("mode", mode, ("logratio", "ratio", "mean"))
        if self.output != "power":
            raise ValueError(
                f"apply_baseline requires output='power', got {self.output!r}."
            )
        out = []
        for ii, data in enumerate(self._data):
            if baseline is None:
                base = data.mean(axis=-1, keepdims=True)
            else:
                times = self.get_times(ii)
                mask = (times >= baseline[0]) & (times <= baseline[1])
                if not mask.any():
                    raise ValueError(
                        f"Baseline {baseline} is empty for epoch {ii}, which "
                        f"spans {times[0]:.3f}-{times[-1]:.3f} s."
                    )
                base = data[..., mask].mean(axis=-1, keepdims=True)
            if mode == "logratio":
                out.append(10 * np.log10(data / base))
            elif mode == "ratio":
                out.append(data / base)
            else:
                out.append(data - base)
        return RaggedEpochsTFR(
            out,
            self.info,
            self.freqs,
            self._tmin,
            output="power",
            alignment=self.alignment,
            events=self.events,
            metadata=self.metadata,
            sfreq=self._sfreq,
        )

    def average(self):
        """Average across epochs.

        Returns
        -------
        data : array, shape (n_channels, n_freqs, n_times)
            The average.

        Raises
        ------
        RaggedTimesError
            If the epochs do not share a time axis. Averaging in that state
            would combine different points of the trial with each other.
        """
        if not self.is_uniform:
            raise RaggedTimesError(
                "Cannot average time-frequency data of unequal duration: a "
                "given time index is not the same point of the trial in every "
                "epoch. Align first, for example with "
                "mne.ragged.warp_tfr(tfr, landmarks, target='median')."
            )
        return np.mean(np.stack(self._data), axis=0)

    def to_mne(self):
        """Convert aligned data to an :class:`mne.time_frequency.EpochsTFR`.

        Returns
        -------
        tfr : instance of EpochsTFRArray
            The equivalent MNE object.
        """
        if not self.is_uniform:
            raise ValueError(
                "Only aligned (uniform) data can be converted; align first."
            )
        info = self.info.copy()
        with info._unlock():
            info["sfreq"] = self._sfreq
        return EpochsTFRArray(
            info=info,
            data=np.stack(self._data),
            times=self.times,
            freqs=self.freqs,
            method="morlet",
        )

    def __repr__(self):
        """Build a string representation of the instance.

        Returns
        -------
        repr : str
            The representation.
        """
        lengths = self.lengths
        span = (
            f"{lengths[0]}"
            if lengths.min() == lengths.max()
            else f"{lengths.min()}-{lengths.max()}"
        )
        alignment = f", aligned: {self.alignment.method}" if self.alignment else ""
        return (
            f"<RaggedEpochsTFR | {len(self)} epochs, {len(self.freqs)} freqs "
            f"({self.freqs[0]:g}-{self.freqs[-1]:g} Hz), {span} samples, "
            f"output={self.output}{alignment}>"
        )


@verbose
def compute_tfr(
    epochs, freqs, *, n_cycles=None, output="power", zero_mean=True, verbose=None
):
    """Compute a Morlet time-frequency representation of ragged epochs.

    Each epoch is transformed at its own duration, so no kernel is modified:
    :func:`mne.time_frequency.tfr_array_morlet` is called once per epoch on an
    ordinary ``(n_channels, n_times)`` array.

    Parameters
    ----------
    epochs : instance of RaggedEpochs
        The epochs.
    freqs : array-like of float
        Frequencies of interest in Hz.
    n_cycles : float | array-like of float | None
        Number of cycles per frequency. Defaults to ``freqs / 2``.
    output : str
        ``'power'``, ``'phase'`` or ``'complex'``.
    zero_mean : bool
        Whether to use zero-mean wavelets.
    %(verbose)s

    Returns
    -------
    tfr : instance of RaggedEpochsTFR
        The time-frequency representation, ragged along time.

    Notes
    -----
    If the epochs carry context (see
    :meth:`mne.ragged.RaggedEpochs.from_raw`), the transform is computed on the
    padded slice and trimmed afterwards, so that no wavelet taper reaches past
    the epoch boundary. For landmark-referenced analyses those boundaries are
    themselves landmarks, so the artifact would otherwise fall exactly where
    the analysis is most sensitive.

    Because every epoch must yield the same frequency axis, the shortest epoch
    determines which frequencies can be computed at all.
    """
    freqs = np.asarray(freqs, dtype=float)
    if n_cycles is None:
        n_cycles = freqs / 2.0
    sfreq = epochs.sfreq

    use_context = epochs.has_context
    n_min = int(epochs.lengths.min())
    if use_context:
        n_min = int((epochs.lengths + epochs.context.sum(axis=1)).min())
    _check_wavelets_fit(freqs, n_cycles, sfreq, n_min)

    out = []
    for ii in range(len(epochs)):
        data = epochs.get_data(ii, with_context=use_context)[np.newaxis]
        tfr = tfr_array_morlet(
            data,
            sfreq=sfreq,
            freqs=freqs,
            n_cycles=n_cycles,
            output=output,
            zero_mean=zero_mean,
            verbose=False,
        )[0]
        if use_context:
            lo, hi = epochs.context[ii]
            if lo or hi:
                tfr = tfr[..., lo : tfr.shape[-1] - hi]
        out.append(tfr)
    logger.info(f"Computed {output} for {len(out)} epochs at {len(freqs)} freqs")
    return RaggedEpochsTFR(
        out,
        epochs.info,
        freqs,
        epochs.tmin,
        output=output,
        events=epochs.events,
        metadata=epochs.metadata,
        sfreq=sfreq,
    )


@verbose
def warp_tfr(
    tfr,
    landmarks,
    *,
    target="median",
    n_points=None,
    landmark_names=None,
    verbose=None,
):
    """Warp a time-frequency representation onto common landmark latencies.

    Parameters
    ----------
    tfr : instance of RaggedEpochsTFR
        The representation to warp.
    landmarks : list of array | str
        Landmark times of each epoch in seconds relative to the epoch start. A
        string is interpreted as the name of a list-valued column of
        ``tfr.metadata``.
    target : str | array-like of float
        The common latencies to warp onto; see
        :func:`mne.ragged.resolve_target_landmarks`.
    n_points : int | None
        Number of samples in the output. Defaults to the median epoch length.
    landmark_names : tuple of str | None
        Names of the landmarks, recorded in the alignment.
    %(verbose)s

    Returns
    -------
    tfr : instance of RaggedEpochsTFR
        Uniform data on the warped axis.

    Notes
    -----
    Energy is moved along the time axis and the frequency axis is left
    untouched, in contrast to warping the time-domain signal before the
    transform, which rescales the frequencies of the oscillations it carries.
    Magnitude is interpolated linearly; when ``output='complex'`` the phase is
    interpolated on the unit circle instead, since linear interpolation of a
    wrapped angle is incorrect.
    """
    if isinstance(landmarks, str):
        if tfr.metadata is None or landmarks not in tfr.metadata:
            raise ValueError(f"tfr.metadata has no column {landmarks!r}.")
        landmarks = [np.asarray(v, dtype=float) for v in tfr.metadata[landmarks]]
    landmarks = [np.asarray(lm, dtype=float) for lm in landmarks]
    if len(landmarks) != len(tfr):
        raise ValueError(f"Got {len(landmarks)} landmark sets for {len(tfr)} epochs.")

    dst = resolve_target_landmarks(landmarks, target)
    if n_points is None:
        n_points = int(np.median(tfr.lengths))
    warp = _warp_complex if np.iscomplexobj(tfr.get_data(0)) else piecewise_linear_warp
    out = [
        warp(tfr.get_data(ii), landmarks[ii], dst, n_points, tfr.sfreq)
        for ii in range(len(tfr))
    ]
    logger.info(
        f"Warped {len(tfr)} epochs onto {n_points} samples spanning "
        f"{dst[0]:.3f}-{dst[-1]:.3f} s"
    )
    record = AlignmentRecord(
        method="piecewise-linear",
        domain="tfr",
        target_coord="seconds",
        original_duration=tfr.lengths / tfr.sfreq,
        original_landmarks=landmarks,
        target_landmarks=dst,
        target_rule=target if isinstance(target, str) else "explicit",
        landmark_names=tuple(landmark_names) if landmark_names else None,
    )
    return RaggedEpochsTFR(
        out,
        tfr.info,
        tfr.freqs,
        float(dst[0]),
        output=tfr.output,
        alignment=record,
        events=tfr.events,
        metadata=tfr.metadata,
        sfreq=(n_points - 1) / (dst[-1] - dst[0]),
    )
