# WWTP Safe-MARL Project Rules

Use these rules whenever continuing this project.

1. Do not train supervised labels on synthetic `load_up` or `rain_dilution` rows; use them for recommendation and robustness tests.
2. Every RL action must pass the safety shield before display or execution.
3. External open data may enrich domain priors and scenario libraries, but must not be mixed into single-plant supervised training unless plant, unit, target, and control semantics are verified.
4. After each major step, update `docs/reflection_log.md` with result, risk, improvement, and next step.
5. Keep raw data, PDFs, DOCX/XLSX files, model binaries, and large generated tables out of GitHub unless explicitly using Git LFS or a release artifact.
6. Dashboard changes must pass `npm run build`; API changes must pass endpoint smoke tests.
