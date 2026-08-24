Variable-duration epochs
========================

:py:mod:`mne.ragged`:

.. automodule:: mne.ragged
   :no-members:
   :no-inherited-members:

Epochs whose trials have different durations, for paradigms where the process
under study lasts as long as it lasts. Trials are stored at their true
durations; analyses that need a common time axis require an explicit alignment
step rather than choosing one implicitly.

.. currentmodule:: mne.ragged

Containers
----------

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/class.rst

   RaggedEpochs
   RaggedEpochsTFR
   AlignmentRecord
   RaggedTimesError

Operations
----------

These are mathematically per-trial and need no common time axis.

.. autosummary::
   :toctree: ../generated/

   apply_baseline
   compute_covariance
   compute_tfr
   concatenate_for_decomposition
   filter_epochs
   map_epochs
   set_eeg_reference

Alignment
---------

Applied explicitly, never implicitly.

.. autosummary::
   :toctree: ../generated/

   align_time
   common_crop
   duration_normalize
   landmark_warp
   pad
   piecewise_linear_warp
   resolve_target_landmarks
   warp_tfr
