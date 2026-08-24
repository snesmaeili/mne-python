"""Tests for epochs of unequal duration."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_allclose

import mne
from mne import create_info
from mne.io import RawArray
from mne.ragged import (
    RaggedEpochs,
    RaggedTimesError,
    apply_baseline,
    common_crop,
    compute_covariance,
    compute_tfr,
    concatenate_for_decomposition,
    duration_normalize,
    filter_epochs,
    landmark_warp,
    pad,
    piecewise_linear_warp,
    resolve_target_landmarks,
    set_eeg_reference,
    warp_tfr,
)

SFREQ = 500.0
F_OSC = 10.0
CH_NAMES = ["C3", "Cz", "C4"]
#: canonical gait landmarks; stance is roughly 60% of the cycle, swing 40%
GAIT_FRACTIONS = np.array([0.00, 0.12, 0.50, 0.62, 1.00])
GAIT_NAMES = ("RHS", "LTO", "LHS", "RTO", "RHS_next")


def _oscillation(duration, sfreq=SFREQ, freq=F_OSC, seed=0):
    """Return a pure oscillation of the requested duration."""
    n_times = int(round(duration * sfreq))
    times = np.arange(n_times) / sfreq
    rng = np.random.default_rng(seed)
    noise = 0.01 * rng.standard_normal(n_times)
    return (np.sin(2 * np.pi * freq * times) + noise)[None, :]


def _peak_frequency(power, freqs):
    """Return the frequency of maximum power over the middle of the epoch."""
    n_times = power.shape[-1]
    core = power[..., n_times // 4 : 3 * n_times // 4].mean(axis=-1)
    return float(freqs[np.argmax(core.ravel())])


@pytest.fixture
def ragged():
    """Return four epochs of unequal duration."""
    info = create_info(CH_NAMES, SFREQ, "eeg")
    rng = np.random.default_rng(1)
    durations = [1.00, 1.24, 1.10, 1.16]
    data = [rng.standard_normal((3, int(d * SFREQ))) * 1e-6 for d in durations]
    return RaggedEpochs(data, info)


@pytest.fixture
def gait():
    """Return gait-like epochs with jittered landmarks."""
    rng = np.random.default_rng(42)
    info = create_info(CH_NAMES, SFREQ, "eeg")
    durations = rng.uniform(0.85, 1.35, 24)
    data, landmarks = [], []
    for duration in durations:
        jitter = np.r_[0.0, rng.normal(0, 0.015, 3), 0.0]
        marks = np.sort(duration * (GAIT_FRACTIONS + jitter))
        marks[0], marks[-1] = 0.0, duration
        n_times = int(round(duration * SFREQ))
        times = np.arange(n_times) / SFREQ
        signal = sum(np.exp(-0.5 * ((times - m) / 0.02) ** 2) for m in marks[1:-1])
        data.append(np.tile(signal, (3, 1)) + 0.01 * rng.standard_normal((3, n_times)))
        landmarks.append(marks)
    return RaggedEpochs(data, info), landmarks


# -- container ------------------------------------------------------------
def test_container_basics(ragged):
    """Test durations, lengths and selection."""
    assert len(ragged) == 4
    assert not ragged.is_uniform
    assert_allclose(ragged.durations, [1.00, 1.24, 1.10, 1.16])
    assert_allclose(ragged.tmax, ragged.durations - 1.0 / SFREQ)

    subset = ragged[[0, 2]]
    assert len(subset) == 2
    assert_allclose(subset.get_data(1), ragged.get_data(2))

    picked = ragged.pick(["C3", "C4"])
    assert picked.ch_names == ["C3", "C4"]
    assert_allclose(picked.get_data(0), ragged.get_data(0)[[0, 2]])


def test_times_raises_rather_than_guessing(ragged):
    """Test that a common time axis is never invented."""
    with pytest.raises(RaggedTimesError, match="do not share a time axis"):
        ragged.times
    # the error must point at the alternatives
    try:
        ragged.times
    except RaggedTimesError as exc:
        for hint in ("durations", "get_times", "align_time"):
            assert hint in str(exc)

    assert ragged.get_times(0)[0] == 0.0
    assert len(ragged.get_times()) == 4


def test_only_time_may_be_ragged():
    """Test that a ragged channel axis is rejected."""
    info = create_info(CH_NAMES, SFREQ, "eeg")
    with pytest.raises(ValueError, match="Only the time axis may be ragged"):
        RaggedEpochs([np.zeros((3, 10)), np.zeros((4, 10))], info)


def test_representations_are_explicit(ragged):
    """Test the three data representations."""
    assert len(ragged.get_data()) == 4
    dense = ragged.get_data(representation="dense")
    assert dense.shape == (4, 3, int(ragged.lengths.max()))
    assert np.isnan(dense[0, 0, -1])
    flat = ragged.get_data(representation="concatenated")
    assert flat.shape == (3, int(ragged.lengths.sum()))
    with pytest.raises(ValueError, match="Invalid value"):
        ragged.get_data(representation="bogus")


def test_uniform_case_matches_epochs_array():
    """Test that equal durations reproduce EpochsArray exactly."""
    info = create_info(CH_NAMES, SFREQ, "eeg")
    rng = np.random.default_rng(11)
    data = rng.standard_normal((6, 3, 250)) * 1e-6
    ragged = RaggedEpochs(list(data), info, tmin=-0.2)
    stock = mne.EpochsArray(data, info, tmin=-0.2, verbose=False)
    assert ragged.is_uniform
    assert_allclose(ragged.times, stock.times, atol=1e-12)
    assert_allclose(ragged.get_data(representation="dense"), stock.get_data())


def test_from_annotations_uses_durations():
    """Test that annotation durations are honored."""
    rng = np.random.default_rng(3)
    raw = RawArray(
        rng.standard_normal((3, 5000)) * 1e-6,
        create_info(CH_NAMES, SFREQ, "eeg"),
        verbose=False,
    )
    onsets = np.arange(1.0, 8.0, 1.0)
    durations = rng.uniform(0.4, 0.9, len(onsets))
    raw.set_annotations(mne.Annotations(onsets, durations, ["trial"] * len(onsets)))
    epochs = RaggedEpochs.from_annotations(raw, description="trial")
    assert len(epochs) == len(onsets)
    assert_allclose(epochs.durations, durations, atol=1.5 / SFREQ)

    raw.set_annotations(
        mne.Annotations(onsets, np.zeros_like(onsets), ["trial"] * len(onsets))
    )
    with pytest.raises(ValueError, match="zero duration"):
        RaggedEpochs.from_annotations(raw, description="trial")


# -- alignment ------------------------------------------------------------
def test_landmarks_coincide_after_warping(gait):
    """Test that warping puts every epoch's landmarks on the target."""
    epochs, landmarks = gait
    dst = resolve_target_landmarks(landmarks, "median")
    n_out = 1000
    grid = np.linspace(dst[0], dst[-1], n_out)

    worst_ms = 0.0
    for ii in range(len(epochs)):
        times = np.arange(epochs.lengths[ii]) / SFREQ
        probe = np.interp(times, landmarks[ii], np.arange(5.0))
        out = piecewise_linear_warp(probe[None], landmarks[ii], dst, n_out, SFREQ)
        recovered = np.interp(dst, grid, out[0])
        err = np.abs(recovered - np.arange(5.0))
        slopes = np.diff(np.arange(5.0)) / np.diff(dst)
        worst_ms = max(worst_ms, (err[1:-1] / slopes[:-1]).max() * 1000)
    # discretization-limited: the error is one source sample
    assert worst_ms < 1.5 * 1000.0 / SFREQ


def test_landmark_error_scales_with_sampling_rate():
    """Test that landmark error is discretization, not bias."""
    rng = np.random.default_rng(3)
    durations = rng.uniform(0.9, 1.3, 8)
    landmarks = [d * GAIT_FRACTIONS for d in durations]
    dst = resolve_target_landmarks(landmarks, "median")
    errors = []
    for sfreq in (250.0, 1000.0):
        worst = 0.0
        for ii, duration in enumerate(durations):
            times = np.arange(int(round(duration * sfreq))) / sfreq
            probe = np.interp(times, landmarks[ii], np.arange(5.0))
            out = piecewise_linear_warp(probe[None], landmarks[ii], dst, 4000, sfreq)[0]
            grid = np.linspace(dst[0], dst[-1], 4000)
            worst = max(worst, np.abs(np.interp(dst, grid, out) - np.arange(5.0)).max())
        errors.append(worst)
    assert errors[1] < errors[0] / 3.0


def test_median_target_preserves_landmark_proportions(gait):
    """Test that the default target keeps the observed proportions."""
    _, landmarks = gait
    median = resolve_target_landmarks(landmarks, "median")
    uniform = resolve_target_landmarks(landmarks, "uniform")

    median_frac = (median - median[0]) / (median[-1] - median[0])
    uniform_frac = (uniform - uniform[0]) / (uniform[-1] - uniform[0])
    assert_allclose(median_frac, GAIT_FRACTIONS, atol=0.03)
    assert_allclose(uniform_frac, np.linspace(0, 1, 5), atol=1e-12)
    # toe-off sits near 12% of the cycle, not 25%
    assert abs(median_frac[1] - 0.12) < 0.03
    assert abs(uniform_frac[1] - 0.25) < 1e-9

    assert_allclose(resolve_target_landmarks(landmarks), median)


def test_mismatched_landmark_counts_rejected():
    """Test that epochs with different landmark counts are refused."""
    landmarks = [np.array([0.0, 0.2, 0.6, 1.0]), np.array([0.0, 0.1, 0.5, 0.7, 1.1])]
    with pytest.raises(ValueError, match="same number of landmarks"):
        resolve_target_landmarks(landmarks)


def test_warp_preserves_durations_and_records_provenance(gait):
    """Test that the original durations survive alignment."""
    epochs, landmarks = gait
    before = epochs.durations.copy()
    aligned = landmark_warp(epochs, landmarks, n_points=600, landmark_names=GAIT_NAMES)
    assert aligned.is_uniform
    record = aligned.alignment
    assert_allclose(record.original_duration, before)
    assert record.original_duration.min() < record.original_duration.max()
    assert record.method == "piecewise-linear"
    assert record.domain == "signal"
    assert record.target_rule == "median"
    assert record.landmark_names == GAIT_NAMES
    assert record.warps_spectral_content


def test_sfreq_is_never_a_phase_axis(gait):
    """Test that sfreq stays a real rate after warping."""
    epochs, landmarks = gait
    n_points = 600
    aligned = landmark_warp(epochs, landmarks, n_points=n_points)
    dst = aligned.alignment.target_landmarks
    assert aligned.sfreq != n_points - 1
    assert_allclose(aligned.sfreq, (n_points - 1) / (dst[-1] - dst[0]))
    assert_allclose(aligned.times[0], dst[0])
    assert_allclose(aligned.times[-1], dst[-1])


def test_common_crop(ragged):
    """Test cropping to the common interval."""
    cropped = common_crop(ragged)
    assert cropped.is_uniform
    assert cropped.lengths[0] == ragged.lengths.min()
    assert_allclose(cropped.alignment.original_duration, ragged.durations)


def test_pad_returns_time_resolved_nave(ragged):
    """Test that padding reports how many epochs contribute over time."""
    padded, nave = pad(ragged)
    assert padded.is_uniform
    assert nave.shape == (int(ragged.lengths.max()),)
    assert nave[0] == len(ragged)
    assert nave[-1] == 1
    assert np.all(np.diff(nave) <= 0)


def test_duration_normalize_is_two_landmark_warp(ragged):
    """Test that duration normalization is the two-landmark case."""
    from_helper = duration_normalize(ragged, n_points=400)
    explicit = landmark_warp(
        ragged, [np.array([0.0, d]) for d in ragged.durations], n_points=400
    )
    for ii in range(len(ragged)):
        assert_allclose(from_helper.get_data(ii), explicit.get_data(ii))
    assert from_helper.alignment.method == "duration-normalize"


# -- time-frequency -------------------------------------------------------
def _two_durations():
    """Return two epochs carrying the same oscillation at different lengths."""
    info = create_info(["Cz"], SFREQ, "eeg")
    return RaggedEpochs([_oscillation(1.0, seed=0), _oscillation(2.0, seed=1)], info)


def test_signal_warp_rescales_apparent_frequency():
    """Test that warping the signal before the transform shifts frequency."""
    epochs = _two_durations()
    freqs = np.arange(4.0, 21.0, 0.5)
    target = np.array([0.0, 1.0])
    n_out = int(SFREQ)
    warped = [
        piecewise_linear_warp(
            epochs.get_data(ii),
            np.array([0.0, epochs.durations[ii]]),
            target,
            n_out,
            SFREQ,
        )
        for ii in range(len(epochs))
    ]
    info = create_info(["Cz"], SFREQ, "eeg")
    tfr = compute_tfr(RaggedEpochs(warped, info), freqs)
    short = _peak_frequency(tfr.get_data(0), freqs)
    long = _peak_frequency(tfr.get_data(1), freqs)
    assert abs(short - F_OSC) < 0.5
    # the stretched epoch is reported near twice its true frequency
    assert abs(long - 2 * F_OSC) < 3.0
    assert abs(long - short) > 5.0


def test_tfr_warp_preserves_frequency():
    """Test that warping the representation leaves frequency alone."""
    epochs = _two_durations()
    freqs = np.arange(4.0, 21.0, 0.5)
    tfr = compute_tfr(epochs, freqs)
    landmarks = [np.array([0.0, d]) for d in epochs.durations]
    aligned = warp_tfr(tfr, landmarks, target="median", n_points=500)
    peaks = [_peak_frequency(aligned.get_data(ii), freqs) for ii in range(2)]
    assert all(abs(p - F_OSC) < 0.5 for p in peaks)
    assert peaks[0] == peaks[1]
    assert not aligned.alignment.warps_spectral_content


def test_tfr_average_requires_alignment():
    """Test that averaging ragged representations is refused."""
    epochs = _two_durations()
    freqs = np.arange(4.0, 21.0, 0.5)
    tfr = compute_tfr(epochs, freqs)
    with pytest.raises(RaggedTimesError, match="Cannot average"):
        tfr.average()
    aligned = warp_tfr(
        tfr, [np.array([0.0, d]) for d in epochs.durations], n_points=500
    )
    assert aligned.average().shape == (1, len(freqs), 500)
    converted = aligned.to_mne()
    assert isinstance(converted, mne.time_frequency.EpochsTFR)
    assert converted.data.shape == (2, 1, len(freqs), 500)


def test_complex_warp_preserves_phase_advance():
    """Test that phase is interpolated on the unit circle."""
    epochs = _two_durations()
    freqs = np.arange(4.0, 21.0, 0.5)
    tfr = compute_tfr(epochs, freqs, output="complex")
    aligned = warp_tfr(
        tfr, [np.array([0.0, d]) for d in epochs.durations], n_points=500
    )
    idx = int(np.argmin(np.abs(freqs - F_OSC)))
    phase = np.angle(aligned.get_data(0)[0, idx])
    diff = np.diff(np.unwrap(phase))
    assert np.median(diff) > 0
    assert (diff > 0).mean() > 0.95


def test_context_protects_epoch_edges():
    """Test that context keeps a wavelet taper off the epoch boundary."""
    sfreq, n_times = 250.0, 6000
    rng = np.random.default_rng(0)
    times = np.arange(n_times) / sfreq
    signal = (
        np.sin(2 * np.pi * F_OSC * times) + 0.1 * rng.standard_normal(n_times)
    ) * 1e-6
    raw = RawArray(signal[None], create_info(["Cz"], sfreq, "eeg"), verbose=False)
    onsets = np.arange(2.0, 20.0, 2.0)
    durations = rng.uniform(1.0, 1.6, len(onsets))
    freqs = np.arange(8.0, 21.0, 1.0)

    bare = RaggedEpochs.from_raw(raw, onsets, durations, context=0.0)
    padded = RaggedEpochs.from_raw(raw, onsets, durations, context=1.0)
    assert not bare.has_context
    assert padded.has_context
    assert_allclose(bare.durations, padded.durations)
    assert_allclose(bare.get_data(0), padded.get_data(0))

    def _edge_ratio(power):
        row = power[0, int(np.argmin(np.abs(freqs - F_OSC)))]
        return row[:12].mean() / row[len(row) // 3 : 2 * len(row) // 3].mean()

    assert _edge_ratio(compute_tfr(bare, freqs).get_data(0)) < 0.6
    assert _edge_ratio(compute_tfr(padded, freqs).get_data(0)) > 0.9


def test_shortest_epoch_bounds_the_frequency_set(ragged):
    """Test that impossible frequency requests are reported clearly."""
    # a fixed n_cycles makes low frequencies expensive, so raising fmin helps
    with pytest.raises(ValueError, match="Raise fmin to at least"):
        compute_tfr(ragged, np.array([2.0, 4.0, 30.0]), n_cycles=12.0)

    # with n_cycles proportional to frequency every wavelet has the same
    # length, so the message must not suggest raising fmin
    freqs = np.array([2.0, 4.0, 30.0])
    with pytest.raises(ValueError, match="raising fmin will not help"):
        compute_tfr(ragged, freqs, n_cycles=freqs)


def test_frequency_check_accounts_for_context():
    """Test that context buys real frequency headroom, not just trimming."""
    rng = np.random.default_rng(5)
    raw = RawArray(
        rng.standard_normal((1, 8000)) * 1e-6,
        create_info(["Cz"], SFREQ, "eeg"),
        verbose=False,
    )
    onsets = np.arange(2.0, 12.0, 2.0)
    durations = np.full(len(onsets), 0.5)
    freqs = np.array([6.0])

    bare = RaggedEpochs.from_raw(raw, onsets, durations, context=0.0)
    with pytest.raises(ValueError, match="longer than the shortest epoch"):
        compute_tfr(bare, freqs, n_cycles=6.0)

    padded = RaggedEpochs.from_raw(raw, onsets, durations, context=1.0)
    tfr = compute_tfr(padded, freqs, n_cycles=6.0)
    assert tfr.lengths[0] == bare.lengths[0]


# -- operations -----------------------------------------------------------
def _per_epoch_reference(epochs, fun):
    """Apply `fun` to each epoch through a stock one-trial EpochsArray."""
    out = []
    for ii in range(len(epochs)):
        array = mne.EpochsArray(
            epochs.get_data(ii)[np.newaxis],
            epochs.info,
            tmin=epochs.tmin[ii],
            verbose=False,
        )
        out.append(fun(array)[0])
    return out


@pytest.mark.filterwarnings("ignore:filter_length.*:RuntimeWarning")
def test_filter_matches_per_epoch_mne(ragged):
    """Test filtering against stock MNE run one epoch at a time."""
    got = filter_epochs(ragged, 1.0, 40.0)
    want = _per_epoch_reference(
        ragged, lambda ea: ea.copy().filter(1.0, 40.0, verbose=False).get_data()
    )
    for ii, expected in enumerate(want):
        assert_allclose(got.get_data(ii), expected, rtol=1e-10, atol=1e-18)


def test_baseline_matches_per_epoch_mne(ragged):
    """Test baseline correction against stock MNE."""
    got = apply_baseline(ragged, baseline=(None, 0.3))
    want = _per_epoch_reference(
        ragged,
        lambda ea: ea.copy().apply_baseline((None, 0.3), verbose=False).get_data(),
    )
    for ii, expected in enumerate(want):
        assert_allclose(got.get_data(ii), expected, rtol=1e-10, atol=1e-18)


def test_average_reference_matches_per_epoch_mne(ragged):
    """Test average referencing against stock MNE."""
    got = set_eeg_reference(ragged, "average")
    want = _per_epoch_reference(
        ragged,
        lambda ea: ea.copy()
        .set_eeg_reference("average", projection=False, verbose=False)
        .get_data(),
    )
    for ii, expected in enumerate(want):
        assert_allclose(got.get_data(ii), expected, rtol=1e-10, atol=1e-18)


def test_tfr_matches_per_epoch_mne(ragged):
    """Test the time-frequency transform against stock MNE."""
    freqs = np.arange(20.0, 41.0, 4.0)
    got = compute_tfr(ragged, freqs, n_cycles=freqs / 2.0)
    want = _per_epoch_reference(
        ragged,
        lambda ea: ea.compute_tfr(
            "morlet",
            freqs=freqs,
            n_cycles=freqs / 2.0,
            return_itc=False,
            average=False,
            verbose=False,
        ).get_data(),
    )
    for ii, expected in enumerate(want):
        assert_allclose(got.get_data(ii), expected, rtol=1e-9)


def test_weighting_policy_is_explicit(ragged):
    """Test that sample and epoch weighting differ and are both available."""
    data, weights = concatenate_for_decomposition(ragged, weighting="samples")
    assert_allclose(data, ragged.get_data(representation="concatenated"))
    assert_allclose(weights, np.ones(len(ragged)))

    _, equal = concatenate_for_decomposition(ragged, weighting="equal")
    contribution = equal * ragged.lengths
    assert_allclose(contribution, contribution[0] * np.ones_like(contribution))

    with pytest.raises(ValueError, match="Invalid value"):
        concatenate_for_decomposition(ragged, weighting="bogus")


def test_covariance_weighting_changes_the_result():
    """Test that the weighting policy affects the covariance."""
    info = create_info(CH_NAMES, SFREQ, "eeg")
    rng = np.random.default_rng(7)
    durations = [1.00, 1.37, 1.12, 0.83, 1.55]
    # variance correlated with duration, so the policy matters
    data = [
        rng.standard_normal((3, int(round(d * SFREQ)))) * (1.0 + 2.0 * k)
        for k, d in enumerate(durations)
    ]
    epochs = RaggedEpochs(data, info)
    by_sample = compute_covariance(epochs, weighting="samples")
    by_epoch = compute_covariance(epochs, weighting="equal")
    assert_allclose(by_sample, by_sample.T, rtol=1e-12)
    relative = np.abs(by_sample - by_epoch).max() / np.abs(by_sample).max()
    assert relative > 0.01
