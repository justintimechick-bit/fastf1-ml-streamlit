# Real model lab validation · November 5, 2026

The real-only interface was published to
<https://fastf1-ml-pit-wall.streamlit.app/> from the existing prepared public F1
repository. Application commit: `dcb4cd34e1596aef374d76025224d99721e9904e`.

Observed validation:

- Default real suite: **29 passed**, including actual real-data fitting, model
  reload, chronological guards, unknown labels, train-only preprocessing, cache
  integrity, UI reruns and release packaging.
- Isolated deployment copy: **25 passed, 4 skipped**. Only the four raw-PDF parsing
  replay cases skip because copyrighted raw evidence caches deliberately remain
  local; all four ran successfully in the workspace suite.
- Scoped Ruff checks and formatting pass. Python source and wheel builds pass.
- Full offline FIA parser replay reproduced all 11 factual/provenance files
  byte-for-byte. Source implementation formatting changed only after capture;
  the original snapshot receipt remains immutable.
- All **1,037 protected archive files** retain their recorded hashes. The original
  `src/f1ml` fingerprint remains
  `3dacc77e40fc73192127b0ad3b0416eb04e92306d8d773d165545acdff67901f`.

Safari checked the hosted Mexico saved forecast (18 earlier training events),
executed a USA historical test (17 earlier events), and verified the race/podium
view, three model explanations and 19-event coverage page. Each forecast retains
22 entrants. All model/task views also passed local real-artifact integration
checks. Normal reruns and identical cached submission were tested with network,
fitting and model deserialization blocked, without changing artifact hashes.

Mexico and USA have separate saved runs in `artifacts/real-lab`; each contains
nine fitted estimators and its own forecast/configuration seal. No setting was
changed to improve a target score. The Mexico executed analysis notebook and
coverage/exclusion reports are included in the release. Runtime-created Cloud
runs are temporary across reboots; use the CLI locally for durable new runs.

These checks establish a functional **2026-only core historical lab**, not the
completion of every original Phase 9 objective. FastF1 timing access remains
restricted, older-season cutoff evidence remains unreviewed, pooled/weighted
comparisons have not run, and no new untouched prospective final test is claimed.
See [measured results](real-model-lab.md) and [data exclusions](real-coverage-20261105.md).
