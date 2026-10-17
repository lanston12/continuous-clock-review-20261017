# Continuous-clock response reconstruction: review artifact

This fixed synthetic-only artifact reproduces the numerical evidence for cell-local temporal compression and shared-field response reconstruction. It contains 24 development and 48 test numerical trajectories from one four-wheel simulator. No proprietary CarSim records, derived outputs, manuscript PDFs, author metadata or credentials are included.

## Environment
Python 3.9; install `python -m pip install -r requirements.txt`. The scientific NumPy, SciPy and Numba versions are pinned.

## Quick check (no simulation)
`python verify_minimal.py`

This verifies all payload SHA-256 hashes, recomputes the four target report counts and the paired wider-prior counts from saved rows, and recomputes the median matched timing ratio. It checks saved evidence, not a fresh simulation or speed measurement.

## Independent numerical replay
`python verify_minimal.py --replay --work-dir /absolute/new/directory`

Copies this snapshot into a fresh external directory, reruns 144 independent raw-dense/compressed sensor/lifting/root checks and 1,728 synthetic raw-dense force queries. It preserves reference inputs and requires identical reporting decisions. A fresh JSON run state records success or failure.

## Complete synthetic regeneration
`python reproduce_public.py --work-dir /absolute/another/new/directory`

Regenerates the 24/48 trajectories, development calibration, main target projections, independent replay, label-mutation controls, eight reused-trajectory prior comparisons and tolerance frontiers. Reference comparison uses abs 1e-7 / rel 1e-8 and exact decisions. No fresh runtime benchmark, proprietary solver, document build or figure generation is claimed. Archived five-repeat timing can be rerun separately with `python code/verify_v7.py benchmark`.

## Expected results and scope
At 750 N, second-minus-first front-window mean-force conditional reports are 31/84/222 of 432 (marginal box / independent windows / shared fields). Other target counts are checked by the verifier. All 1,728 saved targets are included. The eight reused trajectories give 5/20/46, 3/16/28 and 3/16/29 of 72 across three mechanical priors. The archived matched setup-plus-16-direction time-ratio median is 4.63 across 15 configurations.

Compression preserves the native-PL window-response image under independent bounded node errors; sharing fields can narrow directional projections. These are conditional numerical results, not measured-vehicle or population-coverage guarantees. Baseline mechanics are truth-centered; wider priors reuse eight trajectories. Bounds and the 750 N criterion are unchanged.

## Access and rights
Provided for peer review. No open-source redistribution license is granted by this artifact. No rights to third-party simulator material are included. Generic private-data loader functions in shared code are unused and have no accompanying private inputs.
