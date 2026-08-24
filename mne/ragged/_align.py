"""Explicit temporal alignment of ragged epochs."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from collections import Counter

import numpy as np

from ..utils import _check_option, logger, verbose
from ._container import RaggedEpochs
from ._provenance import AlignmentRecord


def piecewise_linear_warp(data, src_landmarks, dst_landmarks, n_out, sfreq):
    """Warp data so that source landmarks fall on target landmarks.

    Parameters
    ----------
    data : array, shape (..., n_times)
        Data to warp along its last axis. Works for time-domain data of shape
        ``(n_channels, n_times)`` and for time-frequency data of shape
        ``(n_channels, n_freqs, n_times)``.
    src_landmarks : array-like of float
        Strictly increasing landmark times in seconds within the epoch,
        beginning at the epoch start and ending at the epoch end.
    dst_landmarks : array-like of float
        Strictly increasing target landmark times in seconds. Must have the
        same length as ``src_landmarks``.
    n_out : int
        Number of samples in the output.
    sfreq : float
        Sampling frequency of ``data`` along its last axis, in Hz.

    Returns
    -------
    warped : array, shape (..., n_out)
        The warped data.

    Notes
    -----
    Implemented as an inverse map: a uniform grid is built in the target
    coordinate, each output sample is mapped back to the source time it comes
    from, and the data is sampled there. This is equivalent to multiplying by a
    linear warp matrix.
    """
    src = np.asarray(src_landmarks, dtype=float)
    dst = np.asarray(dst_landmarks, dtype=float)
    if src.shape != dst.shape:
        raise ValueError(
            f"src_landmarks has shape {src.shape} but dst_landmarks has "
            f"{dst.shape}; they must match."
        )
    if src.size < 2:
        raise ValueError("Need at least a start and an end landmark.")
    if np.any(np.diff(src) <= 0):
        raise ValueError(f"src_landmarks must be strictly increasing, got {src}.")
    if np.any(np.diff(dst) <= 0):
        raise ValueError(f"dst_landmarks must be strictly increasing, got {dst}.")

    data = np.asarray(data)
    t_src = np.arange(data.shape[-1]) / float(sfreq)
    grid = np.linspace(dst[0], dst[-1], n_out)
    t_wanted = np.interp(grid, dst, src)

    flat = data.reshape(-1, data.shape[-1])
    out = np.empty((flat.shape[0], n_out), dtype=flat.dtype)
    for ii in range(flat.shape[0]):
        out[ii] = np.interp(t_wanted, t_src, flat[ii])
    return out.reshape(*data.shape[:-1], n_out)


def _warp_complex(data, src, dst, n_out, sfreq):
    """Warp complex coefficients, treating magnitude and phase separately.

    Parameters
    ----------
    data : array, shape (..., n_times)
        Complex data to warp.
    src : array-like of float
        Source landmark times in seconds.
    dst : array-like of float
        Target landmark times in seconds.
    n_out : int
        Number of samples in the output.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    warped : array, shape (..., n_out)
        The warped complex data.

    Notes
    -----
    Linear interpolation of a wrapped phase angle is incorrect: interpolating
    between +179 and -179 degrees gives 0 rather than 180. The unit complex
    vector is interpolated instead and its argument taken.
    """
    magnitude = piecewise_linear_warp(np.abs(data), src, dst, n_out, sfreq)
    unit = np.exp(1j * np.angle(data))
    real = piecewise_linear_warp(unit.real, src, dst, n_out, sfreq)
    imag = piecewise_linear_warp(unit.imag, src, dst, n_out, sfreq)
    return magnitude * np.exp(1j * np.arctan2(imag, real))


def resolve_target_landmarks(landmarks, target="median"):
    """Choose the common landmark latencies to warp onto.

    Parameters
    ----------
    landmarks : list of array
        Landmark times of each epoch in seconds relative to the epoch start,
        each beginning at the epoch start and ending at the epoch end. All
        entries must have the same length.
    target : str | array-like of float
        ``'median'`` (default) or ``'mean'`` of the observed latencies, or
        ``'uniform'`` to space the landmarks evenly, or an explicit array of
        target latencies. ``'uniform'`` discards the observed proportions
        between landmarks and is rarely appropriate; for gait it would place
        toe-off at 25% of the cycle rather than near 12%.

    Returns
    -------
    target_landmarks : array
        The resolved target latencies, in seconds.
    """
    counts = Counter(len(np.atleast_1d(lm)) for lm in landmarks)
    if len(counts) != 1:
        breakdown = "; ".join(
            f"{n} landmarks: {k} epochs" for n, k in sorted(counts.items())
        )
        raise ValueError(
            f"All epochs must have the same number of landmarks. Got "
            f"{breakdown}. Mapping different numbers of landmarks onto one "
            "target aligns different events with each other. Either drop the "
            "incomplete epochs, or align each group separately."
        )
    stacked = np.asarray(landmarks, dtype=float)
    if isinstance(target, str):
        _check_option("target", target, ("median", "mean", "uniform"))
        if target == "median":
            out = np.median(stacked, axis=0)
        elif target == "mean":
            out = stacked.mean(axis=0)
        else:
            out = np.linspace(0.0, float(np.median(stacked[:, -1])), stacked.shape[1])
    else:
        out = np.asarray(target, dtype=float)
        if out.shape != (stacked.shape[1],):
            raise ValueError(
                f"target has {out.shape[0]} landmarks but the epochs have "
                f"{stacked.shape[1]}."
            )
    if np.any(np.diff(out) <= 0):
        raise ValueError(f"Resolved target landmarks are not increasing: {out}.")
    return out


@verbose
def common_crop(epochs, *, verbose=None):
    """Crop every epoch to the interval present in all of them.

    Parameters
    ----------
    epochs : instance of RaggedEpochs
        The epochs to crop.
    %(verbose)s

    Returns
    -------
    epochs : instance of RaggedEpochs
        Uniform epochs covering the common interval.

    Notes
    -----
    Data beyond the shortest epoch is discarded. This is appropriate when the
    tail of the longer trials is not under study, and inappropriate when the
    variable part is itself the process of interest.
    """
    n_common = int(epochs.lengths.min())
    tmin = float(np.max(epochs.tmin))
    sfreq = epochs.sfreq
    blocks = []
    for ii in range(len(epochs)):
        start = int(round((tmin - epochs.tmin[ii]) * sfreq))
        blocks.append(epochs.get_data(ii)[:, start : start + n_common])
    logger.info(
        f"Cropping to {n_common} samples ({n_common / sfreq:.3f} s), "
        f"discarding up to {epochs.lengths.max() - n_common} samples per epoch"
    )
    record = AlignmentRecord(
        method="common-crop",
        domain="signal",
        target_coord="seconds",
        original_duration=epochs.durations.copy(),
    )
    return RaggedEpochs(
        blocks,
        epochs.info,
        tmin,
        events=epochs.events,
        event_id=epochs.event_id,
        metadata=epochs.metadata,
        alignment=record,
    )


def pad(epochs, pad_value=np.nan):
    """Pad every epoch to the length of the longest one.

    Parameters
    ----------
    epochs : instance of RaggedEpochs
        The epochs to pad. All epochs must share a time origin.
    pad_value : float
        Value used to fill the padded samples.

    Returns
    -------
    epochs : instance of RaggedEpochs
        Uniform epochs, so that ``epochs.times`` is defined.
    nave : array, shape (n_times,)
        Number of epochs contributing a real sample at each time point.

    Notes
    -----
    Under padding the effective number of averaged trials is a function of
    time, which makes the noise level time-dependent and affects any scaling
    that assumes a scalar ``nave``, including the noise covariance used by the
    inverse operator. ``nave`` is returned rather than discarded so that this
    is visible to the caller.
    """
    if not np.allclose(epochs.tmin, epochs.tmin[0]):
        raise ValueError(
            "pad() requires a common time origin, but the epochs have "
            "different tmin. Align them first."
        )
    n_max = int(epochs.lengths.max())
    dense = epochs.get_data(representation="dense", pad_value=pad_value)
    nave = (np.arange(n_max)[None, :] < epochs.lengths[:, None]).sum(axis=0)
    record = AlignmentRecord(
        method="pad",
        domain="signal",
        target_coord="seconds",
        original_duration=epochs.durations.copy(),
    )
    out = RaggedEpochs(
        list(dense),
        epochs.info,
        float(epochs.tmin[0]),
        events=epochs.events,
        event_id=epochs.event_id,
        metadata=epochs.metadata,
        alignment=record,
    )
    return out, nave


def duration_normalize(epochs, n_points=100):
    """Map the start and end of every epoch onto a common axis.

    Parameters
    ----------
    epochs : instance of RaggedEpochs
        The epochs to normalize.
    n_points : int
        Number of samples in the output.

    Returns
    -------
    epochs : instance of RaggedEpochs
        Uniform epochs on the normalized axis.

    Notes
    -----
    This is the two-landmark case of :func:`mne.ragged.landmark_warp`. It
    assumes that the interior of the epoch has no structure worth aligning; if
    landmarks are available, warping to them is preferable.
    """
    landmarks = [np.array([0.0, d]) for d in epochs.durations]
    return _landmark_warp(
        epochs,
        landmarks,
        target="median",
        n_points=n_points,
        landmark_names=("start", "end"),
        method="duration-normalize",
    )


@verbose
def landmark_warp(
    epochs,
    landmarks,
    *,
    target="median",
    n_points=None,
    landmark_names=None,
    verbose=None,
):
    """Warp epochs so that per-trial landmarks coincide.

    Parameters
    ----------
    epochs : instance of RaggedEpochs
        The epochs to warp.
    landmarks : list of array | str
        Landmark times of each epoch in seconds relative to the epoch start,
        each beginning at the epoch start and ending at the epoch end. A string
        is interpreted as the name of a list-valued column of
        ``epochs.metadata``, which is the format produced by
        :meth:`mne.Epochs.add_annotations_to_metadata`.
    target : str | array-like of float
        The common latencies to warp onto; see
        :func:`mne.ragged.resolve_target_landmarks`.
    n_points : int | None
        Number of samples in the output. Defaults to the median epoch length,
        which keeps roughly the native temporal resolution.
    landmark_names : tuple of str | None
        Names of the landmarks, recorded in the alignment.
    %(verbose)s

    Returns
    -------
    epochs : instance of RaggedEpochs
        Uniform epochs on the warped axis.

    Notes
    -----
    This warps the time-domain signal, which rescales the frequencies of the
    oscillations it carries. If the quantity of interest is spectral power,
    compute the time-frequency representation first and warp that instead with
    :func:`mne.ragged.warp_tfr`. The returned object records this in
    ``alignment.warps_spectral_content``.
    """
    return _landmark_warp(
        epochs,
        landmarks,
        target=target,
        n_points=n_points,
        landmark_names=landmark_names,
        method="piecewise-linear",
    )


def _landmark_warp(epochs, landmarks, *, target, n_points, landmark_names, method):
    """Warp epochs onto common landmark latencies.

    Parameters
    ----------
    epochs : instance of RaggedEpochs
        The epochs to warp.
    landmarks : list of array | str
        Landmark times of each epoch, or the name of a metadata column.
    target : str | array-like of float
        The common latencies to warp onto.
    n_points : int | None
        Number of samples in the output.
    landmark_names : tuple of str | None
        Names of the landmarks.
    method : str
        Value recorded as the alignment method.

    Returns
    -------
    epochs : instance of RaggedEpochs
        The warped epochs.
    """
    if isinstance(landmarks, str):
        if epochs.metadata is None or landmarks not in epochs.metadata:
            raise ValueError(f"epochs.metadata has no column {landmarks!r}.")
        landmarks = [np.asarray(v, dtype=float) for v in epochs.metadata[landmarks]]
    landmarks = [np.asarray(lm, dtype=float) for lm in landmarks]
    if len(landmarks) != len(epochs):
        raise ValueError(
            f"Got {len(landmarks)} landmark sets for {len(epochs)} epochs."
        )

    dst = resolve_target_landmarks(landmarks, target)
    if n_points is None:
        n_points = int(np.median(epochs.lengths))
    sfreq = epochs.sfreq
    blocks = [
        piecewise_linear_warp(epochs.get_data(ii), landmarks[ii], dst, n_points, sfreq)
        for ii in range(len(epochs))
    ]

    # The output spans dst[0]..dst[-1] in n_points samples, which is a real
    # constant rate. info["sfreq"] is updated to that rate rather than to a
    # value encoding percent-of-trial.
    info = epochs.info.copy()
    with info._unlock():
        info["sfreq"] = (n_points - 1) / (dst[-1] - dst[0])
    logger.info(
        f"Warped {len(epochs)} epochs onto {n_points} samples spanning "
        f"{dst[0]:.3f}-{dst[-1]:.3f} s"
    )

    record = AlignmentRecord(
        method=method,
        domain="signal",
        target_coord="seconds",
        original_duration=epochs.durations.copy(),
        original_landmarks=landmarks,
        target_landmarks=dst,
        target_rule=target if isinstance(target, str) else "explicit",
        landmark_names=tuple(landmark_names) if landmark_names else None,
    )
    return RaggedEpochs(
        blocks,
        info,
        float(dst[0]),
        events=epochs.events,
        event_id=epochs.event_id,
        metadata=epochs.metadata,
        alignment=record,
    )


def align_time(epochs, method="common-crop", **kwargs):
    """Align epochs onto a common time axis.

    Parameters
    ----------
    epochs : instance of RaggedEpochs
        The epochs to align.
    method : str
        One of ``'common-crop'``, ``'pad'``, ``'duration-normalize'`` or
        ``'landmark'``.
    **kwargs : dict
        Additional keyword arguments passed to the chosen strategy.

    Returns
    -------
    aligned : instance of RaggedEpochs | tuple
        The aligned epochs. ``method='pad'`` additionally returns the
        time-resolved ``nave``.
    """
    _check_option(
        "method", method, ("common-crop", "pad", "duration-normalize", "landmark")
    )
    if method == "common-crop":
        return common_crop(epochs, **kwargs)
    if method == "pad":
        return pad(epochs, **kwargs)
    if method == "duration-normalize":
        return duration_normalize(epochs, **kwargs)
    return landmark_warp(epochs, **kwargs)
