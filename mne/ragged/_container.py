"""Container for epochs of unequal duration."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from .._fiff.meas_info import Info
from .._fiff.pick import _picks_to_idx, pick_info
from ..utils import _check_option, _validate_type, logger, verbose


class RaggedTimesError(RuntimeError):
    """Error raised when epochs do not share a common time axis.

    Raised by :attr:`mne.ragged.RaggedEpochs.times` and by reductions such as
    averaging, which are undefined when trials have different durations.
    """


def _as_index(idx, n):
    """Normalize an index into an integer array.

    Parameters
    ----------
    idx : slice | array-like | int
        The index to normalize.
    n : int
        Length of the axis being indexed.

    Returns
    -------
    idx : array of int
        Integer indices.
    """
    if isinstance(idx, slice):
        return np.arange(n)[idx]
    arr = np.asarray(idx)
    if arr.dtype == bool:
        if arr.shape != (n,):
            raise ValueError(f"Boolean mask has shape {arr.shape}, expected ({n},).")
        return np.flatnonzero(arr)
    return np.atleast_1d(arr).astype(int)


def _validate_blocks(blocks):
    """Check that only the time axis is ragged.

    Parameters
    ----------
    blocks : list of array
        Per-epoch data, each of shape ``(n_channels, n_times_i)``.

    Returns
    -------
    blocks : list of array
        The validated data, cast to float64.
    n_channels : int
        Number of channels, identical across epochs.
    """
    if len(blocks) == 0:
        raise ValueError("Need at least one epoch.")
    out = []
    n_channels = None
    for ii, block in enumerate(blocks):
        block = np.asarray(block, dtype=np.float64)
        if block.ndim != 2:
            raise ValueError(
                f"Epoch {ii} has ndim={block.ndim}, expected 2 "
                "(n_channels, n_times)."
            )
        if n_channels is None:
            n_channels = block.shape[0]
        elif block.shape[0] != n_channels:
            raise ValueError(
                f"Epoch {ii} has {block.shape[0]} channels but epoch 0 has "
                f"{n_channels}. Only the time axis may be ragged."
            )
        if block.shape[1] == 0:
            raise ValueError(f"Epoch {ii} has zero samples.")
        out.append(block)
    return out, int(n_channels)


class RaggedEpochs:
    """Epochs of unequal duration.

    Stores each trial at its own duration. No padding, resampling or time
    warping is applied, and ``info["sfreq"]`` remains the physical sampling
    frequency. Operations that require a common time axis raise
    :class:`~mne.ragged.RaggedTimesError` and name the available alignment
    strategies rather than choosing one.

    Parameters
    ----------
    data : list of array
        Per-epoch data, each of shape ``(n_channels, n_times_i)``. The number
        of channels must be the same for every epoch; only the number of
        samples may differ.
    info : instance of Info
        The measurement info.
    tmin : float | array of float
        Start time of each epoch relative to its event, in seconds. A scalar
        is broadcast to all epochs.
    events : array of int, shape (n_epochs, 3) | None
        The events, in the same format as :class:`mne.Epochs`. If ``None``,
        a trivial event array is generated.
    event_id : dict | None
        Mapping from condition name to integer event code.
    metadata : instance of pandas.DataFrame | None
        Per-epoch metadata, one row per epoch.
    alignment : instance of AlignmentRecord | None
        Record of the temporal alignment applied to produce this object, if
        any. Set by the functions in :mod:`mne.ragged`.
    context : array of int, shape (n_epochs, 2) | None
        Number of samples of surrounding data included in ``data`` on the left
        and right of each epoch. Context is excluded from ``durations``,
        ``lengths`` and :meth:`get_data`, and is used only by transforms that
        would otherwise show an edge artifact at the epoch boundary.

    Attributes
    ----------
    info : instance of Info
        The measurement info.
    events : array of int, shape (n_epochs, 3)
        The events.
    event_id : dict
        Mapping from condition name to integer event code.
    metadata : instance of pandas.DataFrame | None
        Per-epoch metadata.
    alignment : instance of AlignmentRecord | None
        Record of the alignment applied, if any.

    Notes
    -----
    Only the time axis may be ragged. A ragged channel axis is a different
    problem and is not supported here.

    .. versionadded:: 1.12
    """

    def __init__(
        self,
        data,
        info,
        tmin=0.0,
        *,
        events=None,
        event_id=None,
        metadata=None,
        alignment=None,
        context=None,
    ):
        _validate_type(info, Info, "info")
        self._data, n_channels = _validate_blocks(list(data))
        self.info = info
        n_epochs = len(self._data)

        if n_channels != len(info["ch_names"]):
            raise ValueError(
                f"data has {n_channels} channels but info has "
                f"{len(info['ch_names'])}."
            )

        self._lengths = np.array([d.shape[1] for d in self._data], dtype=np.int64)
        if context is None:
            self._context = np.zeros((n_epochs, 2), dtype=np.int64)
        else:
            self._context = np.asarray(context, dtype=np.int64).reshape(n_epochs, 2)
            if np.any(self._context < 0):
                raise ValueError("context must be non-negative.")
            if np.any(self._context.sum(axis=1) >= self._lengths):
                raise ValueError("context leaves no samples in some epochs.")

        self._tmin = np.broadcast_to(np.asarray(tmin, dtype=float), (n_epochs,)).copy()

        if events is None:
            events = np.c_[
                np.arange(n_epochs), np.zeros(n_epochs, int), np.ones(n_epochs, int)
            ]
        self.events = np.asarray(events, dtype=int)
        if self.events.shape != (n_epochs, 3):
            raise ValueError(
                f"events must have shape ({n_epochs}, 3), got {self.events.shape}."
            )
        self.event_id = dict(event_id) if event_id else {"epoch": 1}
        self.metadata = metadata
        self.alignment = alignment

    def __len__(self):
        """Return the number of epochs.

        Returns
        -------
        n_epochs : int
            The number of epochs.
        """
        return len(self._data)

    @property
    def sfreq(self):
        """The physical sampling frequency, in Hz.

        Never rewritten to encode a normalized-phase axis; see
        :class:`~mne.ragged.AlignmentRecord` for how normalization is recorded.
        """
        return float(self.info["sfreq"])

    @property
    def ch_names(self):
        """The channel names."""
        return list(self.info["ch_names"])

    @property
    def lengths(self):
        """Number of samples in each epoch, excluding context."""
        return self._lengths - self._context.sum(axis=1)

    @property
    def durations(self):
        """Duration of each epoch in seconds, excluding context.

        Preserved through alignment, so the original trial durations remain
        available after a warp.
        """
        return self.lengths / self.sfreq

    @property
    def context(self):
        """Samples of surrounding data carried on each side of each epoch."""
        return self._context

    @property
    def has_context(self):
        """Whether any epoch carries surrounding data."""
        return bool(self._context.any())

    @property
    def tmin(self):
        """Start time of each epoch relative to its event, in seconds."""
        return self._tmin

    @property
    def tmax(self):
        """End time of each epoch relative to its event, in seconds."""
        return self._tmin + (self.lengths - 1) / self.sfreq

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
            If the epochs have different durations or different time origins.
            No fallback is applied: returning the shortest common interval
            would silently change the meaning of downstream results.
        """
        if self.is_uniform:
            return self._tmin[0] + np.arange(self.lengths[0]) / self.sfreq
        raise RaggedTimesError(
            f"These {len(self)} epochs do not share a time axis (durations "
            f"{self.durations.min():.3f}-{self.durations.max():.3f} s). "
            "Use one of:\n"
            "    .durations       per-epoch duration in seconds\n"
            "    .get_times(i)    the time vector of one epoch\n"
            "    .get_times()     all time vectors\n"
            "    mne.ragged.align_time(...)  produce a common axis, then .times"
        )

    def get_times(self, epoch=None):
        """Return the time vector of one or all epochs.

        Parameters
        ----------
        epoch : int | None
            Index of the epoch. If ``None``, return a list with one time
            vector per epoch.

        Returns
        -------
        times : array | list of array
            Time vector(s) in seconds.
        """
        if epoch is None:
            return [self.get_times(ii) for ii in range(len(self))]
        return self._tmin[epoch] + np.arange(self.lengths[epoch]) / self.sfreq

    def get_data(
        self,
        epoch=None,
        *,
        representation="ragged",
        pad_value=np.nan,
        with_context=False,
    ):
        """Return the data.

        Parameters
        ----------
        epoch : int | None
            Index of a single epoch. If ``None``, return all epochs in the
            requested representation.
        representation : str
            One of ``'ragged'`` (a list of arrays), ``'dense'`` (a single array
            of shape ``(n_epochs, n_channels, max(lengths))``, right-padded
            with ``pad_value``) or ``'concatenated'`` (a single array of shape
            ``(n_channels, sum(lengths))``). There is no representation that
            suits every caller, so it must be given explicitly.
        pad_value : float
            Value used to pad shorter epochs when
            ``representation='dense'``.
        with_context : bool
            Whether to include the surrounding samples stored as context.

        Returns
        -------
        data : list of array | array
            The requested data.
        """
        _check_option(
            "representation", representation, ("ragged", "dense", "concatenated")
        )
        if epoch is not None:
            return self._epoch_data(epoch, with_context)
        blocks = [self._epoch_data(ii, with_context) for ii in range(len(self))]
        if representation == "ragged":
            return blocks
        if representation == "concatenated":
            return np.concatenate(blocks, axis=1)
        n_max = max(b.shape[1] for b in blocks)
        out = np.full(
            (len(blocks), blocks[0].shape[0], n_max), pad_value, dtype=np.float64
        )
        for ii, block in enumerate(blocks):
            out[ii, :, : block.shape[1]] = block
        return out

    def _epoch_data(self, idx, with_context=False):
        """Return one epoch, optionally including its context.

        Parameters
        ----------
        idx : int
            Index of the epoch.
        with_context : bool
            Whether to include the surrounding samples.

        Returns
        -------
        data : array, shape (n_channels, n_times)
            The epoch data.
        """
        block = self._data[idx]
        if with_context:
            return block
        lo, hi = self._context[idx]
        return block[:, lo : block.shape[1] - hi] if (lo or hi) else block

    def __getitem__(self, item):
        """Select a subset of epochs.

        Parameters
        ----------
        item : int | slice | array-like
            The epochs to select.

        Returns
        -------
        epochs : instance of RaggedEpochs
            The selected epochs.
        """
        sel = _as_index(item, len(self))
        metadata = None
        if self.metadata is not None:
            metadata = self.metadata.iloc[sel].reset_index(drop=True)
        return RaggedEpochs(
            [self._data[ii] for ii in sel],
            self.info,
            self._tmin[sel],
            events=self.events[sel],
            event_id=self.event_id,
            metadata=metadata,
            alignment=self.alignment,
            context=self._context[sel],
        )

    def copy(self):
        """Return a copy of the instance.

        Returns
        -------
        epochs : instance of RaggedEpochs
            A copy.
        """
        return self[np.arange(len(self))]

    def pick(self, picks):
        """Pick a subset of channels.

        Parameters
        ----------
        picks : str | array-like | slice | None
            Channels to include, in any form accepted by
            :func:`mne.pick_types`.

        Returns
        -------
        epochs : instance of RaggedEpochs
            The instance with the selected channels.
        """
        idx = _picks_to_idx(self.info, picks, none="data", exclude=())
        return RaggedEpochs(
            [block[idx] for block in self._data],
            pick_info(self.info, idx, copy=True, verbose=False),
            self._tmin,
            events=self.events,
            event_id=self.event_id,
            metadata=self.metadata,
            alignment=self.alignment,
            context=self._context,
        )

    @classmethod
    @verbose
    def from_raw(
        cls,
        raw,
        onsets,
        durations,
        *,
        tmin=0.0,
        picks=None,
        context=0.0,
        metadata=None,
        event_id=None,
        verbose=None,
    ):
        """Construct epochs of individually specified duration from raw data.

        Parameters
        ----------
        raw : instance of Raw
            The continuous data.
        onsets : array-like of float
            Onset of each epoch in seconds, relative to the start of ``raw``.
        durations : array-like of float
            Duration of each epoch in seconds. Must be the same length as
            ``onsets``.
        tmin : float
            Start of each epoch relative to its onset, in seconds.
        picks : str | array-like | slice | None
            Channels to include, in any form accepted by
            :func:`mne.pick_types`.
        context : float
            Seconds of surrounding data to store alongside each epoch, on both
            sides. Excluded from ``durations``, ``lengths`` and
            :meth:`get_data`. Used by transforms that would otherwise show an
            edge artifact at the epoch boundary, in particular
            :func:`mne.ragged.compute_tfr`, where a wavelet taper reaching past
            the epoch edge would distort the very latency being resolved.
        metadata : instance of pandas.DataFrame | None
            Per-epoch metadata, one row per entry in ``onsets``. Rows for
            dropped epochs are removed.
        event_id : dict | None
            Mapping from condition name to integer event code.
        %(verbose)s

        Returns
        -------
        epochs : instance of RaggedEpochs
            The epochs. Entries that would extend past either end of ``raw``
            are dropped, as they would be by :class:`mne.Epochs`.
        """
        sfreq = raw.info["sfreq"]
        onsets = np.asarray(onsets, dtype=float)
        durations = np.asarray(durations, dtype=float)
        if onsets.shape != durations.shape:
            raise ValueError(
                f"onsets has shape {onsets.shape} but durations has "
                f"{durations.shape}; they must match."
            )
        if np.any(durations <= 0):
            raise ValueError("All durations must be positive.")

        pick_idx = _picks_to_idx(raw.info, picks, none="data", exclude="bads")
        n_context = int(round(context * sfreq))
        n_total = len(raw.times)

        blocks, keep = [], []
        for ii, (onset, duration) in enumerate(zip(onsets, durations)):
            start = int(round((onset + tmin) * sfreq))
            stop = start + int(round((duration - tmin) * sfreq))
            if start - n_context < 0 or stop + n_context > n_total:
                continue
            blocks.append(
                raw.get_data(
                    picks=pick_idx, start=start - n_context, stop=stop + n_context
                )
            )
            keep.append(ii)

        if len(blocks) == 0:
            raise RuntimeError(
                "No epoch fits inside the recording. Check onsets, durations "
                "and context."
            )
        n_dropped = len(onsets) - len(keep)
        if n_dropped:
            logger.info(f"Dropped {n_dropped} epoch(s) extending beyond the data")

        keep = np.asarray(keep)
        if metadata is not None:
            metadata = metadata.iloc[keep].reset_index(drop=True)
        events = np.c_[
            (onsets[keep] * sfreq).astype(int),
            np.zeros(len(keep), int),
            np.ones(len(keep), int),
        ]
        return cls(
            blocks,
            pick_info(raw.info, pick_idx, copy=True, verbose=False),
            tmin,
            events=events,
            event_id=event_id,
            metadata=metadata,
            context=np.tile([n_context, n_context], (len(keep), 1)),
        )

    @classmethod
    @verbose
    def from_annotations(cls, raw, description=None, *, verbose=None, **kwargs):
        """Construct epochs from annotations, using their durations.

        :class:`mne.Epochs` can be constructed around ``raw.annotations.onset``
        but ignores ``raw.annotations.duration``. This constructor uses it.

        Parameters
        ----------
        raw : instance of Raw
            The continuous data, carrying annotations with non-zero duration.
        description : str | list of str | None
            Annotation descriptions to use. If ``None``, all annotations are
            used.
        %(verbose)s
        **kwargs : dict
            Additional keyword arguments passed to
            :meth:`mne.ragged.RaggedEpochs.from_raw`.

        Returns
        -------
        epochs : instance of RaggedEpochs
            The epochs.
        """
        annotations = raw.annotations
        if len(annotations) == 0:
            raise ValueError("raw has no annotations.")
        mask = np.ones(len(annotations), bool)
        if description is not None:
            wanted = {description} if isinstance(description, str) else set(description)
            mask = np.isin(annotations.description, list(wanted))
            if not mask.any():
                raise ValueError(
                    f"No annotation matches description={description!r}. "
                    f"Available: {sorted(set(annotations.description))}"
                )
        if np.any(annotations.duration[mask] <= 0):
            raise ValueError(
                "Some matching annotations have zero duration, so there is no "
                "epoch length to read. Set Annotations.duration, or pass "
                "explicit durations to RaggedEpochs.from_raw()."
            )
        onsets = annotations.onset[mask] - raw.first_time
        return cls.from_raw(raw, onsets, annotations.duration[mask], **kwargs)

    def __repr__(self):
        """Build a string representation of the instance.

        Returns
        -------
        repr : str
            The representation.
        """
        durations = self.durations
        if self.is_uniform:
            span = f"{durations[0]:.3f} s"
        else:
            span = (
                f"{durations.min():.3f}-{durations.max():.3f} s "
                f"(median {np.median(durations):.3f})"
            )
        alignment = f", aligned: {self.alignment.method}" if self.alignment else ""
        return (
            f"<RaggedEpochs | {len(self)} epochs, {len(self.ch_names)} channels, "
            f"{span}, {self.sfreq:g} Hz{alignment}>"
        )
