"""Read-only Streamlit interface over saved f1ml artifacts (Phase 8).

This package deliberately lives beside ``f1ml`` rather than inside it. The Phase 7
lock fingerprints every ``.py`` file under ``src/f1ml`` (``finalization.locking.code_id``);
keeping the interface outside that tree leaves the frozen lock's code fingerprint valid.

The interface never downloads data, fits models, calibrates, selects thresholds or opens
the final-test block. It loads prepared datasets, saved feature matrices and saved model
bundles, and it reuses the pipeline's own prediction guards.
"""
