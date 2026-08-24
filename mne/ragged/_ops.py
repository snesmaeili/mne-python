"""Operations that apply to ragged epochs without a common time axis."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ..filter import filter_data
from ..utils import _check_option, verbose
from ._container import RaggedEpochs


def map_epochs(epochs, fun, *, keep_length=True, **kwargs):
    """Apply a function to each epoch independently.

    Parameters
    ----------
    epochs : instance of RaggedEpochs
        The epochs.
    fun : callable
        Called as ``fun(data, **kwargs)`` with ``data`` of shape
        ``(n_channels, n_times_i)``. Must return an array with the same number
        of channels.
    keep_length : bool
        Whether to require that the output has the same number of samples as
        the input. Set to ``False`` for operations that legitimately change it.
    **kwargs : dict
        Additional keyword arguments passed to ``fun``.

    Returns
    -------
    epochs : instance of RaggedEpochs
        The transformed epochs.

    Notes
    -----
    This is the reference implementation for every operation that is
    mathematically per-trial. It calls existing routines on ordinary
    two-dimensional arrays, so no numerical kernel needs to be modified. The
    cost is a Python-level loop over epochs.
    """
    blocks = []
    for ii in range(len(epochs)):
        data = epochs.get_data(ii)
        out = np.asarray(fun(data, **kwargs))
        if out.shape[0] != data.shape[0]:
            raise ValueError(
                f"fun changed the number of channels of epoch {ii} from "
                f"{data.shape[0]} to {out.shape[0]}."
            )
        if keep_length and out.shape[-1] != data.shape[-1]:
            raise ValueError(
                f"fun changed the length of epoch {ii} from {data.shape[-1]} "
                f"to {out.shape[-1]}. Pass keep_length=False if intended."
            )
        blocks.append(out)
    return RaggedEpochs(
        blocks,
        epochs.info,
        epochs.tmin,
        events=epochs.events,
        event_id=epochs.event_id,
        metadata=epochs.metadata,
        alignment=epochs.alignment,
    )


@verbose
def filter_epochs(epochs, l_freq, h_freq, *, verbose=None, **kwargs):
    """Filter each epoch independently.

    Parameters
    ----------
    epochs : instance of RaggedEpochs
        The epochs.
    l_freq : float | None
        Lower pass-band edge in Hz.
    h_freq : float | None
        Upper pass-band edge in Hz.
    %(verbose)s
    **kwargs : dict
        Additional keyword arguments passed to :func:`mne.filter.filter_data`.

    Returns
    -------
    epochs : instance of RaggedEpochs
        The filtered epochs.

    Notes
    -----
    Filter edge effects depend on epoch duration, so a short and a long epoch
    are not filtered identically. This is also true of fixed-length epochs, but
    there the duration is chosen once. Filtering the continuous data before
    epoching remains preferable where possible.
    """
    kwargs.setdefault("pad", "edge")
    return map_epochs(
        epochs,
        lambda data: filter_data(
            data, epochs.sfreq, l_freq, h_freq, verbose=False, **kwargs
        ),
    )


def apply_baseline(epochs, baseline=(None, 0.0), mode="mean"):
    """Apply a baseline correction to each epoch.

    Parameters
    ----------
    epochs : instance of RaggedEpochs
        The epochs.
    baseline : tuple of float | None
        The baseline window in seconds. ``None`` on either side extends to the
        corresponding edge of the epoch.
    mode : str
        ``'mean'`` to subtract the baseline, ``'ratio'`` to divide by it.

    Returns
    -------
    epochs : instance of RaggedEpochs
        The corrected epochs.
    """
    _check_option("mode", mode, ("mean", "ratio"))

    def _one(data, times):
        lo = times[0] if baseline[0] is None else baseline[0]
        hi = times[-1] if baseline[1] is None else baseline[1]
        mask = (times >= lo) & (times <= hi)
        if not mask.any():
            raise ValueError(
                f"Baseline {baseline} is empty for an epoch spanning "
                f"{times[0]:.3f}-{times[-1]:.3f} s."
            )
        base = data[:, mask].mean(axis=1, keepdims=True)
        return data - base if mode == "mean" else data / base

    blocks = [
        _one(epochs.get_data(ii), epochs.get_times(ii)) for ii in range(len(epochs))
    ]
    return RaggedEpochs(
        blocks,
        epochs.info,
        epochs.tmin,
        events=epochs.events,
        event_id=epochs.event_id,
        metadata=epochs.metadata,
        alignment=epochs.alignment,
    )


def set_eeg_reference(epochs, ref_channels="average"):
    """Re-reference each epoch.

    Parameters
    ----------
    epochs : instance of RaggedEpochs
        The epochs.
    ref_channels : str | list of str
        ``'average'`` for an average reference, or a list of channel names.

    Returns
    -------
    epochs : instance of RaggedEpochs
        The re-referenced epochs.
    """
    if ref_channels == "average":
        return map_epochs(epochs, lambda d: d - d.mean(axis=0, keepdims=True))
    names = epochs.ch_names
    idx = [names.index(ch) for ch in ref_channels]
    return map_epochs(epochs, lambda d: d - d[idx].mean(axis=0, keepdims=True))


def _epoch_weights(epochs, weighting):
    """Return per-epoch weights implementing the requested policy.

    Parameters
    ----------
    epochs : instance of RaggedEpochs
        The epochs.
    weighting : str
        ``'samples'`` or ``'equal'``.

    Returns
    -------
    weights : array, shape (n_epochs,)
        The weights.
    """
    _check_option("weighting", weighting, ("samples", "equal"))
    lengths = epochs.lengths.astype(float)
    if weighting == "samples":
        return np.ones(len(epochs))
    return lengths.mean() / lengths


def concatenate_for_decomposition(epochs, weighting="samples"):
    """Concatenate epochs into a single array for decomposition.

    Parameters
    ----------
    epochs : instance of RaggedEpochs
        The epochs.
    weighting : str
        ``'samples'`` to let every sample count once, so that longer epochs
        contribute proportionally more; ``'equal'`` to give every epoch the
        same total contribution regardless of its duration.

    Returns
    -------
    data : array, shape (n_channels, n_times_total)
        The concatenated data.
    weights : array, shape (n_epochs,)
        The weight applied to each epoch.

    Notes
    -----
    :meth:`mne.preprocessing.ICA.fit` already reshapes :class:`mne.Epochs` to
    ``(n_channels, n_epochs * n_times)``, so decomposition does not require
    equal durations. What unequal durations do require is a choice: with
    fixed-length epochs, weighting by sample and weighting by epoch coincide,
    and they no longer do here. Neither is correct in general, so the weights
    are returned alongside the data rather than being applied silently.
    """
    weights = _epoch_weights(epochs, weighting)
    blocks = [epochs.get_data(ii) * np.sqrt(weights[ii]) for ii in range(len(epochs))]
    return np.concatenate(blocks, axis=1), weights


def compute_covariance(epochs, weighting="samples"):
    """Compute a sensor covariance across epochs of unequal duration.

    Parameters
    ----------
    epochs : instance of RaggedEpochs
        The epochs.
    weighting : str
        ``'samples'`` or ``'equal'``; see
        :func:`mne.ragged.concatenate_for_decomposition`.

    Returns
    -------
    cov : array, shape (n_channels, n_channels)
        The covariance.

    Notes
    -----
    The weighting choice propagates into any noise model built from this
    covariance, including the one used by the inverse operator.
    """
    weights = _epoch_weights(epochs, weighting)
    n_channels = len(epochs.ch_names)
    cov = np.zeros((n_channels, n_channels))
    total = 0.0
    for ii in range(len(epochs)):
        data = epochs.get_data(ii)
        data = data - data.mean(axis=1, keepdims=True)
        cov += weights[ii] * (data @ data.T)
        total += weights[ii] * data.shape[1]
    return cov / total
