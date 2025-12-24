# Repository Guidance for `bid-predictor-dashboard`

## Purpose and Top-Level Layout
- The repo provides visualizations the results of a CatBoost-based bid prediction pipeline.
- Top-level entry points:
  - `dash_app.py`: entrypoint for running the dasboard app.


## Dash UI Helpers
- `dash_app.py` at the repository root should focus on layout and callbacks. Reusable logic lives in the top-level `bid_predictor_ui/` package.
- Each helper module in `bid_predictor_ui/` should stay small and purpose-driven (e.g., data access, formatting, plotting).
- Add or update unit tests under `tests/` whenever changing the UI helpers.
- Keep the dashboard mobile compliant; layout or styling updates should preserve responsive behavior on smaller screens.
- Maintain tab-specific code in separate subdirectories under `bid_predictor_ui/`. Any functionality shared by two or more tabs should live at the top level of `bid_predictor_ui/`.


## Testing Strategy
- Unit tests live under `tests/unit/`.
- Run the full suite with `pytest -q` from the repo root.
- The `--testing` CLI flag mirrors the integration suite's small evaluation split; use it when running scripts in CI-like contexts.

## Contribution Tips
- Maintain deterministic behavior: keep random seeds plumbed from CLI into CatBoost and CV splitters.
- Normalize any user-facing or serialized outputs (NumPy -> Python types) to prevent downstream errors.
