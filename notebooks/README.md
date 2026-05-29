# Notebooks

Exploration **only**. Per CLAUDE.md §6:

- Do **not** import from notebooks into `src/` (production code).
- Naming: `NN_short_topic.ipynb` — numbered for ordering.
- Strip output cells before committing: `nbstripout` (installed with `[dev]`).

Suggested first notebooks (from the planning docs):

| File | Topic |
|---|---|
| `01_dataset_overview.ipynb` | Explore the external email-classifier dataset; inspect its raw label set to inform the 4-class mapping. |
| `02_error_analysis.ipynb` | Per-class errors + confusion matrix once a baseline exists. |
