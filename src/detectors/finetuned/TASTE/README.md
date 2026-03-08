# TASTE Integration Notes

This folder keeps the migrated TASTE implementation and resources under:

- `taste.py`
- `translation/*-English.json`
- package init file

The translation dictionaries are auto-loaded from this folder by default:

- default path: `src/detectors/finetuned/TASTE/translation`
- code anchor: `_DEFAULT_DICT_DIR = str(Path(__file__).parent / "translation")`

## How TASTE Is Registered In `mgt_eval`

1. Keep TASTE source under `src/detectors/finetuned/TASTE/`.
2. Ensure detector import path is reachable and loaded by `detectors` package init.
3. Keep `@register("taste")` and `@register_train("taste")` in `taste.py`.
4. Add train adapter mapping for `taste` in CLI so unified `train` args route to TASTE trainer args.
5. Add detect/train YAML examples for `taste`.
6. Add detector metadata entry (`src/detector_metadata/detectors.json`) for UI/CLI discoverability.
7. Add TASTE checkpoint artifact resolution in CLI detect flow so `run_dir/final` and `epoch_*` can be consumed directly.

## Evaluating TASTE Training Artifacts

Supported inputs in detect mode:

- direct HF checkpoint dir (contains `config.json`)
- TASTE run root dir (auto-resolve to `final/`, fallback latest `epoch_*`)
- explicit `model_path` (from YAML or CLI)

Example:

```bash
mgteval-cli detect examples/detect/taste.yaml
```
